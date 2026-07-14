import os
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple
import warnings
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augment import get_augmentor
from .pyramid import PyramidDataset
from .utils import save_distilled_bundle


@dataclass
class LGMConfig:
    ipc: int = 2
    lr: float = 2e-3
    iterations: int = 5000
    augs_per_batch: int = 10
    distill_mode: str = 'pyramid'
    aug_mode: str = 'standard'
    decorrelate_color: bool = True
    init_mode: str = 'noise'
    pyramid_extent_it: int = 200
    pyramid_start_res: int = 1
    image_log_it: int = 200
    loss_log_it: int = 500
    checkpoint_it: int = 200
    syn_res: int = 256
    crop_res: int = 224
    noise_std: float = 0.0
    save_every: int = 200
    resume: bool = True
    real_batch_cap: int = 64
    syn_batch_cap: int = 40
    max_grad_norm: float = 1.0
    # Semantic-drift constraints: class-mean anchoring and
    # Mahalanobis pairwise covariance calibration (Lcov).
    feature_mean_weight: float = 0.0
    lcov_weight: float = 0.0
    patch_weight: float = 0.0
    patch_sample_k: int = 0
    stat_batch_size: int = 64
    diversity_weight: float = 0.01
    diversity_margin: float = 0.8
    match_ipc: int = 1
    gm_target: str = 'linear'
    adapter_last_blocks: int = 1
    adapter_include_head: bool = True
    adapter_include_lora: bool = True
    adapter_grad_chunk_cap: int = 2
    adaptive_enable: bool = False
    adaptive_rounds: int = 0
    adaptive_eval_max_per_class: int = 64
    adaptive_eval_use_testset: bool = True
    adaptive_eval_batch_size: int = 100
    adaptive_eval_epochs: int = 1000
    adaptive_eval_lr: float = 1e-3
    adaptive_eval_weight_decay: float = 0.0
    adaptive_eval_patience: int = 50
    adaptive_eval_log_interval: int = 50
    adaptive_ridge: float = 1e-3
    adaptive_drop_margin: float = 2.0
    adaptive_low_margin: float = 5.0
    adaptive_min_class_acc: float = 70.0
    adaptive_extra_ipc: int = 1
    adaptive_max_ipc: int = 3
    adaptive_extra_iterations: int = 600
    adaptive_focus_only: bool = True
    adaptive_save_best: bool = True


class _FC(nn.Module):
    def __init__(self, num_feats: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(num_feats, num_classes, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LinearGMForMACIL:
    """Official LGM logic ported into the MACIL project.

    Differences from the official repo are limited to project integration:
    - backbone access is through an injected feature_fn
    - persistence uses local filesystem helpers instead of wandb loggers
    - dataset normalization is identity because MACIL's current training path
      already feeds tensors directly to the ViT without the official repo's
      dataset.normalize() wrapper
    """

    def __init__(
        self,
        cfg: LGMConfig,
        feature_fn: Callable[[Tensor], Tensor],
        num_feats: int,
        num_classes: int,
        train_dataset,
        device: torch.device,
        log_dir: str,
        label_offset: int = 0,
        token_fn: Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]] = None,
        model: Optional[nn.Module] = None,
        task_id: Optional[int] = None,
        test_dataset=None,
        feature_means: Optional[Tensor] = None,
        class_invcovs: Optional[Tensor] = None,
        reference_features_by_class: Optional[list] = None,
    ):
        self.cfg = cfg
        self.feature_fn = feature_fn
        self.token_fn = token_fn
        self.model = model
        self.task_id = task_id
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.log_dir = log_dir
        self.label_offset = label_offset
        self.global_step = 0
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        suggested_real_bs = self._effective_match_ipc() * self.cfg.augs_per_batch * self.num_classes
        self.real_batch_size = max(1, min(len(self.train_dataset), suggested_real_bs, int(self.cfg.real_batch_cap)))
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=0,
            batch_size=self.real_batch_size,
            pin_memory=True,
            drop_last=False,
        )
        self.train_iter = iter(self.train_loader)

        self.distilled_dataset = PyramidDataset(
            num_classes=self.num_classes,
            cfg=self.cfg,
            device=self.device,
            out_dir=self.log_dir,
        )
        self.syn_augmentor = get_augmentor(aug_mode=self.cfg.aug_mode, crop_res=self.cfg.crop_res, noise_std=self.cfg.noise_std)
        self.real_augmentor = get_augmentor(aug_mode=self.cfg.aug_mode, crop_res=self.cfg.crop_res, noise_std=self.cfg.noise_std)
        self.class_patch_templates = None
        self.class_feature_means = feature_means.detach().cpu() if torch.is_tensor(feature_means) else None
        self.class_feature_invcovs = class_invcovs.detach().cpu() if torch.is_tensor(class_invcovs) else None
        self.reference_features_by_class = [f.detach().cpu() if torch.is_tensor(f) else None for f in reference_features_by_class] if reference_features_by_class is not None else None
        if (self.cfg.patch_weight > 0 or
                self.cfg.feature_mean_weight > 0 or self.cfg.lcov_weight > 0):
            self._precompute_class_targets()
        self.focus_class_ids = None
        self._eval_cache = None
        self.adaptive_history = []
        self.best_probe_macro = float('-inf')
        self.best_probe_class_acc = None
        self.best_synset_state = None
        if self.cfg.resume:
            self.load_checkpoint()

        self.adapter_params = self._select_adapter_params()
        if self.adapter_params:
            for param in self.adapter_params:
                param.requires_grad_(True)
            logging.info(
                '[Official-LGM] Adapter-aware GM enabled with %d trainable target tensors '
                '(last_blocks=%s, include_head=%s, include_lora=%s).',
                len(self.adapter_params),
                getattr(self.cfg, 'adapter_last_blocks', 1),
                getattr(self.cfg, 'adapter_include_head', True),
                getattr(self.cfg, 'adapter_include_lora', True),
            )


    def _optimizer_params(self):
        params = []
        for group in self.distilled_dataset.optimizer.param_groups:
            params.extend(group.get('params', []))
        return params

    def _use_adapter_gm(self) -> bool:
        return str(getattr(self.cfg, 'gm_target', 'linear')).lower() in ('adapter', 'adapter_aware', 'continual')

    def _select_adapter_params(self):
        if not self._use_adapter_gm() or self.model is None:
            return []
        task_id = self.task_id
        if task_id is None:
            task_id = int(getattr(self.model, 'numtask', 1)) - 1
        task_id = max(0, int(task_id))
        last_blocks = max(1, int(getattr(self.cfg, 'adapter_last_blocks', 1)))
        block_start = max(0, 12 - last_blocks)
        include_head = bool(getattr(self.cfg, 'adapter_include_head', True))
        include_lora = bool(getattr(self.cfg, 'adapter_include_lora', True))
        params = []
        seen = set()
        for name, param in self.model.named_parameters():
            use = False
            if include_head and f'classifier_pool.{task_id}' in name:
                use = True
            if include_lora and 'image_encoder.blocks.' in name and any(tok in name for tok in ('lora_', 'glora_', 'elora_')):
                parts = name.split('.')
                try:
                    block_idx = int(parts[parts.index('blocks') + 1])
                except (ValueError, IndexError):
                    block_idx = -1
                if block_idx >= block_start:
                    if f'.{task_id}.' in name or 'glora_' in name:
                        use = True
            if use and id(param) not in seen:
                params.append(param)
                seen.add(id(param))
        if not params:
            logging.warning('[Official-LGM] gm_target=adapter requested but no adapter params were selected; falling back to linear GM.')
        return params

    @staticmethod
    def _flatten_param_grads(params, grads):
        flats = []
        for param, grad in zip(params, grads):
            if grad is None:
                flats.append(torch.zeros_like(param, memory_format=torch.contiguous_format).view(-1))
            else:
                flats.append(grad.contiguous().view(-1))
        if not flats:
            return None
        return torch.cat(flats, dim=0)

    def _effective_match_ipc(self) -> int:
        match_ipc = int(getattr(self.cfg, 'match_ipc', 0) or 0)
        if match_ipc <= 0:
            return int(self.cfg.ipc)
        return max(1, min(int(self.cfg.ipc), match_ipc))

    def _select_match_indices(self, y_syn: Tensor) -> Tensor:
        match_ipc = self._effective_match_ipc()
        if match_ipc <= 0 or y_syn.numel() == 0:
            return torch.arange(y_syn.size(0), device=y_syn.device)
        picked = []
        unique_labels = y_syn.unique(sorted=True)
        if self.focus_class_ids is not None:
            focus = {int(c) for c in self.focus_class_ids}
            unique_labels = torch.tensor([int(c.item()) for c in unique_labels if int(c.item()) in focus], device=y_syn.device, dtype=y_syn.dtype)
            if unique_labels.numel() == 0:
                unique_labels = y_syn.unique(sorted=True)
        for c in unique_labels.tolist():
            cls_idx = torch.nonzero(y_syn == int(c), as_tuple=False).flatten()
            if cls_idx.numel() == 0:
                continue
            take = min(int(match_ipc), int(cls_idx.numel()))
            slot0 = int(max(0, self.global_step - 1)) % int(cls_idx.numel())
            offsets = (torch.arange(take, device=y_syn.device) + slot0) % int(cls_idx.numel())
            picked.append(cls_idx.index_select(0, offsets))
        if not picked:
            return torch.arange(y_syn.size(0), device=y_syn.device)
        return torch.cat(picked, dim=0)

    def _extract_cls_with_grad_chunked(self, x: Tensor, with_grad: bool = True) -> Tensor:
        feats = []
        syn_cap = max(1, int(self.cfg.syn_batch_cap))
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            for start in range(0, x.size(0), syn_cap):
                end = min(start + syn_cap, x.size(0))
                x_chunk = x[start:end]
                if self.token_fn is not None:
                    cls_chunk, _ = self.token_fn(x_chunk)
                else:
                    cls_chunk = self.feature_fn(x_chunk)
                feats.append(cls_chunk)
        return torch.cat(feats, dim=0) if feats else x.new_empty((0, self.num_feats))

    def _same_class_diversity_loss(self, feats: Tensor, labels: Tensor) -> Tensor:
        if feats.numel() == 0 or labels.numel() <= 1:
            return feats.new_zeros(())
        counts = torch.bincount(labels, minlength=self.num_classes)
        if int((counts > 1).sum().item()) <= 0:
            return feats.new_zeros(())
        feats = nn.functional.normalize(feats, p=2, dim=1)
        loss = feats.new_zeros(())
        groups = 0
        for c in labels.unique(sorted=True):
            idx = torch.nonzero(labels == c, as_tuple=False).flatten()
            if idx.numel() <= 1:
                continue
            f = feats.index_select(0, idx)
            sim = f @ f.t()
            mask = ~torch.eye(f.size(0), dtype=torch.bool, device=f.device)
            off_diag = sim[mask]
            if off_diag.numel() == 0:
                continue
            loss = loss + F.relu(off_diag - float(self.cfg.diversity_margin)).mean()
            groups += 1
        if groups == 0:
            return feats.new_zeros(())
        return loss / float(groups)

    @staticmethod
    def _grads_are_finite(params) -> bool:
        for param in params:
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False
        return True

    def _bundle_metadata(self):
        class_ipc = self.distilled_dataset.get_class_ipc() if hasattr(self.distilled_dataset, 'get_class_ipc') else [int(self.cfg.ipc)] * int(self.num_classes)
        return dict(
            ipc=int(self.cfg.ipc),
            class_ipc=[int(x) for x in class_ipc],
            num_classes=int(self.num_classes),
            gm_target=str(getattr(self.cfg, 'gm_target', 'linear')),
            adapter_last_blocks=int(getattr(self.cfg, 'adapter_last_blocks', 1)),
            adaptive_history=self.adaptive_history,
        )

    def _save_bundle_at(self, out_dir: Path):
        with torch.no_grad():
            syn_images, syn_labels = self.distilled_dataset.get_data()
        save_distilled_bundle(
            syn_images.detach(),
            syn_labels.detach(),
            out_dir,
            step=self.global_step,
            **self._bundle_metadata(),
        )

    def _run_distill_steps(self, start_step: int, end_step: int, desc: str = 'Distilling Images'):
        if start_step > end_step:
            return
        for i in tqdm(
            range(start_step, end_step + 1),
            initial=self.global_step,
            total=end_step,
            desc=desc,
        ):
            self.global_step = i
            _ = self.match_gradients()
            self.distilled_dataset.upkeep(step=self.global_step)

    def _extract_feature_chunked(self, x: Tensor, with_grad: bool = False) -> Tensor:
        feats = []
        batch_cap = max(1, int(self.cfg.syn_batch_cap))
        ctx = nullcontext() if with_grad else torch.no_grad()
        with ctx:
            for start in range(0, x.size(0), batch_cap):
                end = min(start + batch_cap, x.size(0))
                x_chunk = x[start:end]
                if self.token_fn is not None:
                    cls_chunk, _ = self.token_fn(x_chunk)
                else:
                    cls_chunk = self.feature_fn(x_chunk)
                feats.append(cls_chunk)
        return torch.cat(feats, dim=0) if feats else x.new_empty((0, self.num_feats))

    def _build_eval_cache(self):
        if self._eval_cache is not None:
            return self._eval_cache
        use_testset = bool(getattr(self.cfg, 'adaptive_eval_use_testset', True)) and self.test_dataset is not None
        eval_dataset = self.test_dataset if use_testset else self.train_dataset
        max_per_class = 0 if use_testset else int(getattr(self.cfg, 'adaptive_eval_max_per_class', 0) or 0)
        batch_size = max(1, min(256, int(getattr(self.cfg, 'adaptive_eval_batch_size', self.cfg.stat_batch_size))))
        loader = DataLoader(
            eval_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=False,
        )
        feats_by_class = [[] for _ in range(self.num_classes)]
        counts = [0 for _ in range(self.num_classes)]
        for _, images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long() - self.label_offset
            feats = self._extract(images)
            for c in labels.unique(sorted=True).tolist():
                c = int(c)
                if c < 0 or c >= self.num_classes:
                    continue
                cls_idx = torch.nonzero(labels == c, as_tuple=False).flatten()
                if cls_idx.numel() == 0:
                    continue
                remain = cls_idx.numel()
                if max_per_class > 0:
                    remain = min(remain, max(0, max_per_class - counts[c]))
                if remain <= 0:
                    continue
                pick = cls_idx[:remain]
                feats_by_class[c].append(feats.index_select(0, pick).detach().cpu())
                counts[c] += int(remain)
            if max_per_class > 0 and all(v >= max_per_class for v in counts):
                break
        feat_list, label_list = [], []
        for c in range(self.num_classes):
            if not feats_by_class[c]:
                continue
            cur = torch.cat(feats_by_class[c], dim=0)
            feat_list.append(cur)
            label_list.append(torch.full((cur.size(0),), c, dtype=torch.long))
        if feat_list:
            real_feats = torch.cat(feat_list, dim=0).float()
            real_labels = torch.cat(label_list, dim=0).long()
        else:
            real_feats = torch.empty((0, self.num_feats), dtype=torch.float32)
            real_labels = torch.empty((0,), dtype=torch.long)
        self._eval_cache = (real_feats, real_labels)
        return self._eval_cache

    def _score_classifier(self, classifier: nn.Module, eval_feats: Tensor, eval_labels: Tensor):
        classifier.eval()
        class_acc = []
        correct_total = 0
        total = 0
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = eval_labels == c
                denom = int(mask.sum().item())
                if denom <= 0:
                    class_acc.append(0.0)
                    continue
                feats_c = eval_feats[mask].to(self.device, non_blocking=True)
                logits = classifier(feats_c)
                preds = logits.argmax(dim=1).cpu()
                labels_c = eval_labels[mask].cpu()
                correct = int((preds == labels_c).sum().item())
                class_acc.append(100.0 * correct / float(denom))
                correct_total += correct
                total += denom
        macro_acc = float(sum(class_acc) / max(1, len(class_acc)))
        overall_acc = 100.0 * float(correct_total) / max(1, total)
        return dict(class_acc=class_acc, macro_acc=macro_acc, overall_acc=overall_acc)

    def _fit_linear_probe_from_synset(self, eval_feats: Tensor, eval_labels: Tensor):
        batch_size = max(1, int(getattr(self.cfg, 'adaptive_eval_batch_size', 100)))
        epochs = max(1, int(getattr(self.cfg, 'adaptive_eval_epochs', 1000)))
        lr = float(getattr(self.cfg, 'adaptive_eval_lr', 1e-3))
        weight_decay = float(getattr(self.cfg, 'adaptive_eval_weight_decay', 0.0))
        patience = max(1, int(getattr(self.cfg, 'adaptive_eval_patience', 50)))
        log_interval = max(1, int(getattr(self.cfg, 'adaptive_eval_log_interval', 50)))

        with torch.no_grad():
            syn_images, syn_labels = self.distilled_dataset.get_data()
            syn_images = syn_images.detach()
            syn_labels = syn_labels.detach().long()
        if syn_labels.numel() == 0:
            class_acc = [0.0 for _ in range(self.num_classes)]
            return dict(class_acc=class_acc, macro_acc=0.0, overall_acc=0.0, best_epoch=0)

        classifier = torch.nn.Linear(self.num_feats, self.num_classes, bias=True).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max(1, epochs))

        best_metrics = None
        best_overall = float('-inf')
        best_epoch = 0
        stale_epochs = 0

        num_syn = syn_labels.size(0)
        for epoch in range(1, epochs + 1):
            classifier.train()
            perm = torch.randperm(num_syn, device=syn_labels.device)
            epoch_loss = 0.0
            seen = 0
            for start in range(0, num_syn, batch_size):
                end = min(start + batch_size, num_syn)
                idx = perm[start:end]
                batch_images = syn_images.index_select(0, idx)
                batch_labels = syn_labels.index_select(0, idx)
                aug_images = self.syn_augmentor(batch_images)
                batch_feats = self._extract_feature_chunked(aug_images, with_grad=False)
                logits = classifier(batch_feats)
                loss = F.cross_entropy(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().item()) * int(batch_labels.size(0))
                seen += int(batch_labels.size(0))
            scheduler.step()

            metrics = self._score_classifier(classifier=classifier, eval_feats=eval_feats, eval_labels=eval_labels)
            overall_acc = float(metrics['overall_acc'])
            improved = overall_acc > best_overall + 1e-8
            if improved:
                best_overall = overall_acc
                best_epoch = epoch
                best_metrics = metrics
                stale_epochs = 0
            else:
                stale_epochs += 1
            if epoch == 1 or epoch % log_interval == 0 or epoch == epochs or stale_epochs >= patience:
                logging.info(
                    f'[Official-LGM-AdaptiveEval] epoch={epoch}/{epochs} '
                    f'loss={epoch_loss / max(1, seen):.4f} overall={overall_acc:.2f} '
                    f'best={best_overall:.2f} stale={stale_epochs}/{patience}'
                )
            if stale_epochs >= patience:
                break

        if best_metrics is None:
            best_metrics = self._score_classifier(classifier=classifier, eval_feats=eval_feats, eval_labels=eval_labels)
            best_epoch = min(1, epochs)
        best_metrics = dict(best_metrics)
        best_metrics['best_epoch'] = int(best_epoch)
        return best_metrics

    def _evaluate_synset_classwise(self):
        eval_feats, eval_labels = self._build_eval_cache()
        if eval_labels.numel() == 0:
            class_acc = [0.0 for _ in range(self.num_classes)]
            return dict(class_acc=class_acc, macro_acc=0.0, overall_acc=0.0, best_epoch=0)
        return self._fit_linear_probe_from_synset(eval_feats=eval_feats, eval_labels=eval_labels)

    def _snapshot_current_synset(self):
        return {
            'synset': self.distilled_dataset.get_save_dict(),
            'global_step': int(self.global_step),
            'history': list(self.adaptive_history),
        }

    def _restore_synset_snapshot(self, snapshot: dict):
        if snapshot is None:
            return
        self.distilled_dataset.load_from_dict(snapshot['synset'])
        self.global_step = int(snapshot.get('global_step', self.global_step))
        self.adaptive_history = list(snapshot.get('history', self.adaptive_history))

    def _select_adaptive_classes(self, class_acc, round_idx: int):
        if not class_acc:
            return []
        mean_acc = float(sum(class_acc) / max(1, len(class_acc)))
        low_thr = max(float(self.cfg.adaptive_min_class_acc), mean_acc - float(self.cfg.adaptive_low_margin))
        weak = set(i for i, acc in enumerate(class_acc) if float(acc) < low_thr)
        if self.best_probe_class_acc is not None:
            for i, acc in enumerate(class_acc):
                if float(acc) < float(self.best_probe_class_acc[i]) - float(self.cfg.adaptive_drop_margin):
                    weak.add(i)
        max_ipc = int(getattr(self.cfg, 'adaptive_max_ipc', self.cfg.ipc))
        current_ipc = self.distilled_dataset.get_class_ipc() if hasattr(self.distilled_dataset, 'get_class_ipc') else [int(self.cfg.ipc)] * self.num_classes
        weak = [int(i) for i in sorted(weak) if int(current_ipc[int(i)]) < max_ipc]
        return weak

    def _run_adaptive_boost(self):
        if not bool(getattr(self.cfg, 'adaptive_enable', False)):
            return
        max_rounds = max(0, int(getattr(self.cfg, 'adaptive_rounds', 0)))
        if max_rounds <= 0:
            return
        for round_idx in range(max_rounds + 1):
            metrics = self._evaluate_synset_classwise()
            class_acc = [float(x) for x in metrics['class_acc']]
            macro_acc = float(metrics['macro_acc'])
            overall_acc = float(metrics['overall_acc'])
            best_epoch = int(metrics.get('best_epoch', 0))
            current_ipc = self.distilled_dataset.get_class_ipc() if hasattr(self.distilled_dataset, 'get_class_ipc') else [int(self.cfg.ipc)] * self.num_classes
            round_info = {
                'round': int(round_idx),
                'step': int(self.global_step),
                'macro_acc': macro_acc,
                'overall_acc': overall_acc,
                'best_epoch': best_epoch,
                'class_acc': class_acc,
                'class_ipc': [int(x) for x in current_ipc],
            }
            eval_source = 'test' if bool(getattr(self.cfg, 'adaptive_eval_use_testset', True)) and self.test_dataset is not None else 'train_proxy'
            logging.info(f'[Official-LGM-Adaptive] round={round_idx} step={self.global_step} eval={eval_source} best_epoch={best_epoch} macro={macro_acc:.2f} overall={overall_acc:.2f} class_ipc={current_ipc}')
            self.adaptive_history.append(round_info)
            if macro_acc >= self.best_probe_macro:
                self.best_probe_macro = macro_acc
                self.best_probe_class_acc = list(class_acc)
                self.best_synset_state = self._snapshot_current_synset()
            weak_classes = self._select_adaptive_classes(class_acc, round_idx=round_idx)
            round_info['weak_classes'] = [int(x) for x in weak_classes]
            if weak_classes:
                logging.info(f'[Official-LGM-Adaptive] weak_classes={weak_classes}')
            if not weak_classes or round_idx >= max_rounds:
                break
            boosted = self.distilled_dataset.expand_classes(
                weak_classes,
                extra_ipc=int(getattr(self.cfg, 'adaptive_extra_ipc', 1)),
                max_ipc=int(getattr(self.cfg, 'adaptive_max_ipc', self.cfg.ipc)),
            )
            round_info['boosted_classes'] = [int(x) for x in boosted]
            if boosted:
                logging.info(f'[Official-LGM-Adaptive] boosted_classes={boosted} -> class_ipc={self.distilled_dataset.get_class_ipc()}')
            extra_steps = max(0, int(getattr(self.cfg, 'adaptive_extra_iterations', 0)))
            if extra_steps <= 0:
                continue
            self.focus_class_ids = set(boosted) if bool(getattr(self.cfg, 'adaptive_focus_only', True)) and boosted else None
            start_step = int(self.global_step) + 1
            end_step = int(self.global_step) + extra_steps
            self._run_distill_steps(start_step, end_step, desc=f'Adaptive Distill R{round_idx + 1}')
            self.focus_class_ids = None
        if bool(getattr(self.cfg, 'adaptive_save_best', True)) and self.best_synset_state is not None:
            self._restore_synset_snapshot(self.best_synset_state)

    def distill(self):
        start_step = int(self.global_step) + 1
        if start_step <= int(self.cfg.iterations):
            self._run_distill_steps(start_step, int(self.cfg.iterations), desc='Distilling Images')
        self._run_adaptive_boost()

        if bool(getattr(self.cfg, 'save_final_checkpoint', False)):
            self.save_checkpoint()
        # Keep only the final distilled images: canonical root bundle for replay
        # plus an explicit final/ bundle for inspection. No iter_* or latest bundles.
        self._save_bundle_at(Path(self.log_dir) / 'final')
        self.save_data()

    def match_gradients(self):
        AMP_SCALE = 1024.0
        x_real, y_real = self.get_real_batch()
        x_syn, y_syn = self.distilled_dataset.get_data()

        if self._use_adapter_gm() and self.adapter_params:
            grad_real = self.get_adapter_real_grad(x_real=x_real, y_real=y_real)
            grad_syn, feature_mean_loss, lcov_loss, patch_loss, div_loss = self.get_adapter_syn_grad(x_syn=x_syn, y_syn=y_syn)
        else:
            fc = _FC(num_feats=self.num_feats, num_classes=self.num_classes).to(self.device)
            grad_real = self.get_real_grad(x_real=x_real, y_real=y_real, fc=fc)
            grad_syn, feature_mean_loss, lcov_loss, patch_loss, div_loss = self.get_syn_grad(x_syn=x_syn, y_syn=y_syn, fc=fc)

        match_loss = 1 - torch.nn.functional.cosine_similarity(grad_real, grad_syn, dim=0)
        total_loss = match_loss
        if feature_mean_loss is not None:
            total_loss = total_loss + float(self.cfg.feature_mean_weight) * feature_mean_loss
        if lcov_loss is not None:
            total_loss = total_loss + float(self.cfg.lcov_weight) * lcov_loss
        if patch_loss is not None:
            total_loss = total_loss + float(self.cfg.patch_weight) * patch_loss
        if div_loss is not None:
            total_loss = total_loss + float(self.cfg.diversity_weight) * div_loss
        if not torch.isfinite(total_loss.detach()):
            warnings.warn(f'[Official-LGM] Non-finite distill loss at step {self.global_step}; skipping optimizer step.')
            try:
                self.distilled_dataset.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.distilled_dataset.optimizer.zero_grad()
            return float('nan')
        scaled_loss = total_loss * AMP_SCALE
        try:
            self.distilled_dataset.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.distilled_dataset.optimizer.zero_grad()
        scaled_loss.backward()
        params = self._optimizer_params()
        for param in params:
            if param.grad is not None:
                param.grad.div_(AMP_SCALE)
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(self.cfg.max_grad_norm))
        if not self._grads_are_finite(params):
            warnings.warn(f'[Official-LGM] Non-finite synset gradients at step {self.global_step}; skipping optimizer step.')
            try:
                self.distilled_dataset.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.distilled_dataset.optimizer.zero_grad()
            return float(total_loss.detach().item())
        self.distilled_dataset.optimizer.step()
        if self.model is not None:
            try:
                self.model.zero_grad(set_to_none=True)
            except TypeError:
                self.model.zero_grad()
        feature_mean_val = 0.0 if feature_mean_loss is None else float(feature_mean_loss.detach().item())
        lcov_val = 0.0 if lcov_loss is None else float(lcov_loss.detach().item())
        patch_val = 0.0 if patch_loss is None else float(patch_loss.detach().item())
        div_val = 0.0 if div_loss is None else float(div_loss.detach().item())
        self.last_loss_terms = {
            'total': float(total_loss.detach().item()),
            'gm': float(match_loss.detach().item()),
            'gm_w': float(match_loss.detach().item()),
            'feature_mean': feature_mean_val,
            'feature_mean_w': float(self.cfg.feature_mean_weight) * feature_mean_val,
            'lcov': lcov_val,
            'lcov_w': float(self.cfg.lcov_weight) * lcov_val,
            'patch': patch_val,
            'patch_w': float(self.cfg.patch_weight) * patch_val,
            'div': div_val,
            'div_w': float(self.cfg.diversity_weight) * div_val,
        }
        log_it = max(1, int(getattr(self.cfg, 'loss_log_it', getattr(self.cfg, 'image_log_it', 200)) or 200))
        if self.global_step % log_it == 0 or self.global_step == int(self.cfg.iterations):
            logging.info(
                '[Official-LGM-Detail] Step %d/%d => Total %.4f | GM %.4f (w %.4f) | Mean %.4f * %.4f = %.4f | Lcov %.4f * %.4f = %.4f | Patch %.4f * %.4f = %.4f | Div %.4f * %.4f = %.4f',
                int(self.global_step), int(self.cfg.iterations),
                self.last_loss_terms['total'],
                self.last_loss_terms['gm'], self.last_loss_terms['gm_w'],
                self.last_loss_terms['feature_mean'], float(self.cfg.feature_mean_weight), self.last_loss_terms['feature_mean_w'],
                self.last_loss_terms['lcov'], float(self.cfg.lcov_weight), self.last_loss_terms['lcov_w'],
                self.last_loss_terms['patch'], float(self.cfg.patch_weight), self.last_loss_terms['patch_w'],
                self.last_loss_terms['div'], float(self.cfg.diversity_weight), self.last_loss_terms['div_w'],
            )
        return float(total_loss.detach().item())

    def get_real_batch(self) -> Tuple[Tensor, Tensor]:
        tries = 8 if self.focus_class_ids is not None else 1
        last_batch = None
        for _ in range(max(1, tries)):
            batch_real = next(self.train_iter, None)
            if batch_real is None:
                self.train_iter = iter(self.train_loader)
                batch_real = next(self.train_iter, None)
            _, x_real, y_real = batch_real
            x_real = x_real.to(self.device, non_blocking=True)
            y_real = y_real.to(self.device, non_blocking=True).long() - self.label_offset
            last_batch = (x_real, y_real)
            if self.focus_class_ids is None:
                return x_real, y_real
            focus_mask = torch.zeros_like(y_real, dtype=torch.bool)
            for cls in self.focus_class_ids:
                focus_mask |= (y_real == int(cls))
            if bool(focus_mask.any()):
                return x_real[focus_mask], y_real[focus_mask]
        return last_batch

    def _extract(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            z = self.feature_fn(x)
        return z


    def _extract_tokens_no_grad(self, x: Tensor):
        if self.token_fn is not None:
            with torch.no_grad():
                cls, patch = self.token_fn(x)
            return cls, patch
        return self._extract(x), None

    def _precompute_class_targets(self):
        batch_size = max(1, min(int(self.cfg.stat_batch_size), len(self.train_dataset)))
        loader = DataLoader(
            self.train_dataset,
            shuffle=False,
            num_workers=0,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=False,
        )
        feat_sum = torch.zeros(self.num_classes, self.num_feats, device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)
        collect_ref_feats = (self.reference_features_by_class is None) and float(getattr(self.cfg, 'lcov_weight', 0.0)) > 0
        class_feat_chunks = [[] for _ in range(self.num_classes)] if collect_ref_feats else None
        patch_sum = None
        for _, images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long() - self.label_offset
            cls, patch = self._extract_tokens_no_grad(images)
            valid = (labels >= 0) & (labels < self.num_classes)
            if not valid.any():
                continue
            cls = cls[valid]
            labels = labels[valid]
            if patch is not None:
                patch = patch[valid]
                if patch_sum is None:
                    patch_sum = torch.zeros(self.num_classes, patch.size(1), patch.size(2), device=self.device)
            for c in labels.unique():
                c_int = int(c.item())
                mask = labels == c
                cls_c = cls[mask]
                feat_sum[c_int] += cls_c.sum(dim=0)
                counts[c_int] += float(mask.sum().item())
                if class_feat_chunks is not None:
                    class_feat_chunks[c_int].append(cls_c.detach().cpu())
                if patch_sum is not None:
                    patch_sum[c_int] += patch[mask].sum(dim=0)
        counts = counts.clamp_min(1.0)
        means = feat_sum / counts.unsqueeze(1)
        if self.class_feature_means is None:
            self.class_feature_means = means.detach().cpu()
        if class_feat_chunks is not None:
            self.reference_features_by_class = []
            invcovs = []
            eye = torch.eye(self.num_feats, device=self.device)
            for c in range(self.num_classes):
                if class_feat_chunks[c]:
                    ref = torch.cat(class_feat_chunks[c], dim=0).float()
                    self.reference_features_by_class.append(ref.cpu())
                    ref_dev = ref.to(self.device)
                    if ref_dev.size(0) > 1:
                        cov = torch.cov(ref_dev.T)
                    else:
                        cov = eye.clone()
                    cov = cov + eye * 1e-3
                    invcovs.append(torch.linalg.pinv(cov).detach().cpu())
                else:
                    self.reference_features_by_class.append(None)
                    invcovs.append(eye.detach().cpu())
            if self.class_feature_invcovs is None:
                self.class_feature_invcovs = torch.stack(invcovs, dim=0)
        if patch_sum is not None:
            self.class_patch_templates = (patch_sum / counts.view(-1, 1, 1)).detach().cpu()

    @staticmethod
    def _angle_weighted_patch_loss(new_patch: Tensor, target_patch: Tensor, new_cls: Tensor, patch_sample_k: int = 0) -> Tensor:
        new_patch_norm = nn.functional.normalize(new_patch, p=2, dim=-1)
        target_patch_norm = nn.functional.normalize(target_patch, p=2, dim=-1)
        alpha_cos = nn.functional.cosine_similarity(new_cls.unsqueeze(1), new_patch, dim=-1).clamp(min=-1.0, max=1.0)
        alpha_angle = 1 - (torch.acos(alpha_cos) / torch.pi)
        distances = torch.norm(new_patch_norm - target_patch_norm, p=2, dim=-1)
        weights = (1 - alpha_angle.detach())
        if patch_sample_k > 0 and patch_sample_k < weights.size(1):
            top_idx = torch.topk(weights, k=patch_sample_k, dim=1, largest=True).indices
            distances = torch.gather(distances, 1, top_idx)
            weights = torch.gather(weights, 1, top_idx)
        return (weights * distances).mean()

    @staticmethod
    def _pairwise_mahalanobis(feats: Tensor, invcov: Tensor) -> Tensor:
        if feats.size(0) <= 1:
            return feats.new_zeros((0,))
        diff = feats.unsqueeze(1) - feats.unsqueeze(0)
        dist2 = torch.einsum('...d,df,...f->...', diff, invcov, diff).clamp_min(0.0)
        dist = torch.sqrt(dist2 + 1e-8)
        tri = torch.triu_indices(feats.size(0), feats.size(0), offset=1, device=feats.device)
        return dist[tri[0], tri[1]]

    def _semantic_drift_alignment_loss(self, feats: Tensor, labels: Tensor):
        if feats.numel() == 0 or labels.numel() == 0:
            zero = feats.new_zeros(())
            return zero, zero
        mean_loss = feats.new_zeros(())
        lcov_loss = feats.new_zeros(())
        mean_groups = 0
        cov_groups = 0
        for c in labels.unique(sorted=True):
            idx = torch.nonzero(labels == c, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            c_int = int(c.item())
            if c_int < 0 or c_int >= self.num_classes:
                continue
            cls_feats = feats.index_select(0, idx)
            if self.class_feature_means is not None:
                target_mean = self.class_feature_means[c_int].to(self.device, dtype=cls_feats.dtype)
                mean_loss = mean_loss + F.mse_loss(cls_feats.mean(dim=0), target_mean)
                mean_groups += 1
            if self.class_feature_invcovs is not None and cls_feats.size(0) > 1:
                invcov = self.class_feature_invcovs[c_int].to(self.device, dtype=cls_feats.dtype)
                syn_dist = self._pairwise_mahalanobis(cls_feats, invcov)
                ref_dist = None
                if self.reference_features_by_class is not None and c_int < len(self.reference_features_by_class):
                    ref_feats = self.reference_features_by_class[c_int]
                    if torch.is_tensor(ref_feats) and ref_feats.size(0) > 1:
                        ref_feats = ref_feats.to(self.device, dtype=cls_feats.dtype)
                        take = min(ref_feats.size(0), cls_feats.size(0))
                        # Deterministic rotating subset keeps the target pair count aligned with current IPC.
                        offset = int(max(0, self.global_step - 1)) % int(ref_feats.size(0))
                        ref_idx = (torch.arange(take, device=self.device) + offset) % int(ref_feats.size(0))
                        ref_dist = self._pairwise_mahalanobis(ref_feats.index_select(0, ref_idx), invcov).detach()
                if ref_dist is not None and ref_dist.numel() > 0 and syn_dist.numel() > 0:
                    n = min(int(ref_dist.numel()), int(syn_dist.numel()))
                    lcov_loss = lcov_loss + F.l1_loss(syn_dist[:n], ref_dist[:n])
                    cov_groups += 1
                elif syn_dist.numel() > 0:
                    # Fallback when reference pair features are unavailable: keep the
                    # average pairwise Mahalanobis distance near the Gaussian target.
                    target = torch.full_like(syn_dist, float((2.0 * self.num_feats) ** 0.5))
                    lcov_loss = lcov_loss + F.l1_loss(syn_dist, target)
                    cov_groups += 1
        if mean_groups > 0:
            mean_loss = mean_loss / float(mean_groups)
        if cov_groups > 0:
            lcov_loss = lcov_loss / float(cov_groups)
        return mean_loss, lcov_loss

    def get_real_grad(self, x_real: Tensor, y_real: Tensor, fc: nn.Module) -> Tensor:
        x_real = x_real.detach()
        y_real = y_real.detach()
        with autocast(enabled=False):
            x_real = self.real_augmentor(x_real)
            z_real = self._extract(x_real)
            out_real = fc(z_real)
            loss_real = nn.functional.cross_entropy(out_real, y_real)
            grad_real_w, grad_real_b = torch.autograd.grad(loss_real, [fc.linear.weight, fc.linear.bias], retain_graph=False, create_graph=False)
            grad_real = torch.cat([grad_real_w.detach().flatten(), grad_real_b.detach().flatten()], dim=0)
        return grad_real

    def _adapter_logits(self, images: Tensor) -> Tensor:
        if self.model is None:
            raise RuntimeError('Adapter-aware GM requires a model reference.')
        out = self.model(images)
        if isinstance(out, dict):
            return out['logits']
        return out

    def get_adapter_real_grad(self, x_real: Tensor, y_real: Tensor) -> Tensor:
        x_real = x_real.detach()
        y_real = y_real.detach()
        with autocast(enabled=False):
            x_real = self.real_augmentor(x_real)
            logits = self._adapter_logits(x_real)
            loss_real = nn.functional.cross_entropy(logits, y_real)
            grads = torch.autograd.grad(
                loss_real,
                self.adapter_params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
        grad_real = self._flatten_param_grads(self.adapter_params, grads)
        if grad_real is None:
            raise RuntimeError('Adapter-aware GM could not compute real gradients.')
        return grad_real.detach()

    def get_adapter_syn_grad(self, x_syn: Tensor, y_syn: Tensor):
        grad_syn = None
        patch_loss_acc = x_syn.new_zeros(()) if self.cfg.patch_weight > 0 and self.class_patch_templates is not None and self.token_fn is not None else None
        div_loss = None
        feature_mean_loss = None
        lcov_loss = None
        clean_feats = None
        need_clean_feats = (bool(float(self.cfg.diversity_weight) > 0) or
                            bool(float(self.cfg.feature_mean_weight) > 0) or
                            bool(float(self.cfg.lcov_weight) > 0))
        if need_clean_feats:
            clean_feats = self._extract_cls_with_grad_chunked(x_syn, with_grad=True)
        if clean_feats is not None and float(self.cfg.diversity_weight) > 0 and int((torch.bincount(y_syn, minlength=self.num_classes) > 1).sum().item()) > 0:
            div_loss = self._same_class_diversity_loss(clean_feats, y_syn)
        if clean_feats is not None and (float(self.cfg.feature_mean_weight) > 0 or float(self.cfg.lcov_weight) > 0):
            feature_mean_loss, lcov_loss = self._semantic_drift_alignment_loss(clean_feats, y_syn)

        match_indices = self._select_match_indices(y_syn)
        x_match = x_syn.index_select(0, match_indices)
        y_match = y_syn.index_select(0, match_indices)

        syn_cap = max(1, int(getattr(self.cfg, 'adapter_grad_chunk_cap', self.cfg.syn_batch_cap)))
        syn_cap = min(syn_cap, max(1, int(self.cfg.syn_batch_cap)))
        for _ in range(self.cfg.augs_per_batch):
            x_aug = self.syn_augmentor(x_match)
            y_aug = y_match
            n = x_aug.size(0)
            for start in range(0, n, syn_cap):
                end = min(start + syn_cap, n)
                x_chunk = x_aug[start:end]
                y_chunk = y_aug[start:end]
                with autocast(enabled=False):
                    logits = self._adapter_logits(x_chunk)
                    ce_syn = nn.functional.cross_entropy(logits, y_chunk)
                    grads = torch.autograd.grad(
                        ce_syn,
                        self.adapter_params,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )
                    chunk_grad = self._flatten_param_grads(self.adapter_params, grads)
                    if patch_loss_acc is not None:
                        cls_syn, patch_syn = self.token_fn(x_chunk)
                        patch_targets = self.class_patch_templates[y_chunk.detach().cpu()].to(self.device, dtype=patch_syn.dtype)
                        patch_loss_acc = patch_loss_acc + self._angle_weighted_patch_loss(
                            patch_syn, patch_targets, cls_syn, patch_sample_k=int(self.cfg.patch_sample_k)
                        )
                if chunk_grad is None:
                    continue
                if grad_syn is None:
                    grad_syn = chunk_grad
                else:
                    grad_syn = grad_syn + chunk_grad
        if grad_syn is None:
            raise RuntimeError('Adapter-aware GM could not compute synthetic gradients.')
        aug_div = float(self.cfg.augs_per_batch)
        grad_syn = grad_syn / aug_div
        if patch_loss_acc is not None:
            patch_loss_acc = patch_loss_acc / aug_div
        return grad_syn, feature_mean_loss, lcov_loss, patch_loss_acc, div_loss

    def get_syn_grad(self, x_syn: Tensor, y_syn: Tensor, fc: nn.Module):
        grad_syn = None
        patch_loss_acc = x_syn.new_zeros(()) if self.cfg.patch_weight > 0 and self.class_patch_templates is not None and self.token_fn is not None else None
        div_loss = None
        feature_mean_loss = None
        lcov_loss = None
        clean_feats = None
        need_clean_feats = (bool(float(self.cfg.diversity_weight) > 0) or
                            bool(float(self.cfg.feature_mean_weight) > 0) or
                            bool(float(self.cfg.lcov_weight) > 0))
        if need_clean_feats:
            clean_feats = self._extract_cls_with_grad_chunked(x_syn, with_grad=True)
        if clean_feats is not None and float(self.cfg.diversity_weight) > 0 and int((torch.bincount(y_syn, minlength=self.num_classes) > 1).sum().item()) > 0:
            div_loss = self._same_class_diversity_loss(clean_feats, y_syn)
        if clean_feats is not None and (float(self.cfg.feature_mean_weight) > 0 or float(self.cfg.lcov_weight) > 0):
            feature_mean_loss, lcov_loss = self._semantic_drift_alignment_loss(clean_feats, y_syn)

        match_indices = self._select_match_indices(y_syn)
        x_match = x_syn.index_select(0, match_indices)
        y_match = y_syn.index_select(0, match_indices)

        syn_cap = max(1, int(self.cfg.syn_batch_cap))
        for _ in range(self.cfg.augs_per_batch):
            x_aug = self.syn_augmentor(x_match)
            y_aug = y_match
            n = x_aug.size(0)
            for start in range(0, n, syn_cap):
                end = min(start + syn_cap, n)
                x_chunk = x_aug[start:end]
                y_chunk = y_aug[start:end]
                with autocast(enabled=False):
                    if self.token_fn is not None and (self.cfg.patch_weight > 0 and self.class_patch_templates is not None):
                        cls_syn, patch_syn = self.token_fn(x_chunk)
                        z_syn = cls_syn
                    else:
                        z_syn = self.feature_fn(x_chunk)
                        cls_syn, patch_syn = z_syn, None
                    out_syn = fc(z_syn)
                    ce_syn = nn.functional.cross_entropy(out_syn, y_chunk)
                    grad_syn_w, grad_syn_b = torch.autograd.grad(
                        ce_syn,
                        [fc.linear.weight, fc.linear.bias],
                        retain_graph=True,
                        create_graph=True,
                    )
                    chunk_grad = torch.cat([grad_syn_w.flatten(), grad_syn_b.flatten()], dim=0)
                    if patch_loss_acc is not None and patch_syn is not None:
                        patch_targets = self.class_patch_templates[y_chunk.detach().cpu()].to(self.device, dtype=patch_syn.dtype)
                        patch_loss_acc = patch_loss_acc + self._angle_weighted_patch_loss(
                            patch_syn, patch_targets, cls_syn, patch_sample_k=int(self.cfg.patch_sample_k)
                        )
                if grad_syn is None:
                    grad_syn = chunk_grad
                else:
                    grad_syn = grad_syn + chunk_grad
        aug_div = float(self.cfg.augs_per_batch)
        grad_syn = grad_syn / aug_div
        if patch_loss_acc is not None:
            patch_loss_acc = patch_loss_acc / aug_div
        return grad_syn, feature_mean_loss, lcov_loss, patch_loss_acc, div_loss

    def save_data(self):
        with torch.no_grad():
            syn_images, syn_labels = self.distilled_dataset.get_data()
            save_distilled_bundle(
                syn_images.detach(),
                syn_labels.detach(),
                Path(self.log_dir),
                step=self.global_step,
                **self._bundle_metadata(),
            )

    def load_checkpoint(self):
        ckpt = Path(self.log_dir) / 'ckpt.pth'
        if ckpt.exists():
            load_dict = torch.load(ckpt, map_location='cpu')
            self.distilled_dataset.load_from_dict(load_dict['synset'])
            self.global_step = int(load_dict['global_step'])
            self.adaptive_history = list(load_dict.get('adaptive_history', []))
            self.best_probe_macro = float(load_dict.get('best_probe_macro', self.best_probe_macro))
            self.best_probe_class_acc = load_dict.get('best_probe_class_acc', self.best_probe_class_acc)
            best_synset = load_dict.get('best_synset', None)
            if best_synset is not None:
                best_step = load_dict.get('best_synset_global_step', self.global_step)
                if best_step is None:
                    best_step = self.global_step
                best_history = load_dict.get('best_synset_history', self.adaptive_history)
                if best_history is None:
                    best_history = self.adaptive_history
                self.best_synset_state = {
                    'synset': best_synset,
                    'global_step': int(best_step),
                    'history': list(best_history),
                }
            torch.set_rng_state(load_dict['random_state']['torch'])
            if torch.cuda.is_available() and load_dict['random_state']['cuda'] is not None:
                torch.cuda.set_rng_state_all(load_dict['random_state']['cuda'])
            random.setstate(load_dict['random_state']['python'])
            np.random.set_state(load_dict['random_state']['numpy'])

    def save_checkpoint(self):
        random_state = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'python': random.getstate(),
            'numpy': np.random.get_state(),
        }
        save_dict = {
            'synset': self.distilled_dataset.get_save_dict(),
            'global_step': self.global_step,
            'random_state': random_state,
            'adaptive_history': self.adaptive_history,
            'best_probe_macro': self.best_probe_macro,
            'best_probe_class_acc': self.best_probe_class_acc,
            'best_synset': None if self.best_synset_state is None else self.best_synset_state.get('synset'),
            'best_synset_global_step': None if self.best_synset_state is None else self.best_synset_state.get('global_step'),
            'best_synset_history': None if self.best_synset_state is None else self.best_synset_state.get('history'),
        }
        tmp = Path(self.log_dir) / 'tmp.pth'
        torch.save(save_dict, tmp)
        os.replace(tmp, Path(self.log_dir) / 'ckpt.pth')
