from dataclasses import dataclass
import logging
import gc
import shutil
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .augment import get_augmentor
from .linear_gm import _FC
from .pyramid import PyramidDataset
from .utils import save_distilled_bundle


@dataclass
class RefreshConfig:
    lr: float = 1e-3
    iterations: int = 300
    augs_per_batch: int = 1
    aug_mode: str = 'standard'
    image_log_it: int = 100
    loss_log_it: int = 100
    checkpoint_it: int = 100
    save_every: int = 100
    crop_res: int = 224
    noise_std: float = 0.0
    syn_batch_cap: int = 2
    grad_weight: float = 1.0
    image_anchor_weight: float = 0.05
    kd_weight: float = 0.1
    temperature: float = 2.0
    max_grad_norm: float = 1.0
    precompute_batch_cap: int = 8
    class_subsample: int = 0
    class_chunk_size: int = 4
    # Semantic-drift constraints: class-mean anchoring and
    # Mahalanobis pairwise covariance calibration (Lcov).
    feature_mean_weight: float = 0.0
    lcov_weight: float = 0.0
    patch_weight: float = 0.05
    patch_sample_k: int = 0
    shared_aug_refresh: bool = True
    projector_weight: float = 0.5
    projector_mix_alpha: float = 0.7


class SynsetRefreshForMACIL:
    def __init__(
        self,
        cfg: RefreshConfig,
        old_feature_fn: Optional[Callable[[Tensor], Tensor]],
        new_feature_fn: Optional[Callable[[Tensor], Tensor]],
        num_feats: int,
        num_classes: int,
        device: torch.device,
        task_dir: str,
        refresh_tag: str,
        old_logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
        new_logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
        old_token_fn: Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]] = None,
        new_token_fn: Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]] = None,
        drift_projector_fn: Optional[Callable[[Tensor], Tensor]] = None,
        feature_means: Optional[Tensor] = None,
        class_invcovs: Optional[Tensor] = None,
    ):
        self.cfg = cfg
        self.old_feature_fn = old_feature_fn
        self.new_feature_fn = new_feature_fn
        self.old_logits_fn = old_logits_fn
        self.new_logits_fn = new_logits_fn
        self.old_token_fn = old_token_fn
        self.new_token_fn = new_token_fn
        self.drift_projector_fn = drift_projector_fn
        self.num_feats = num_feats
        self.num_classes = num_classes
        self.device = device
        self.task_dir = Path(task_dir)
        self.refresh_dir = self.task_dir / 'refresh_history' / refresh_tag
        self.syn_augmentor = get_augmentor(aug_mode=self.cfg.aug_mode, crop_res=self.cfg.crop_res, noise_std=self.cfg.noise_std)
        self.distilled_dataset = self._load_synset()
        for group in self.distilled_dataset.optimizer.param_groups:
            group['lr'] = float(self.cfg.lr)
        self.reference_images, self.reference_labels = self.distilled_dataset.get_data()
        self.reference_images = self.reference_images.detach().cpu()
        self.reference_labels = self.reference_labels.detach().cpu()
        self.feature_means = feature_means.detach().cpu() if torch.is_tensor(feature_means) else None
        self.class_invcovs = class_invcovs.detach().cpu() if torch.is_tensor(class_invcovs) else None
        if (self.feature_means is None and float(getattr(self.cfg, 'feature_mean_weight', 0.0)) > 0) or \
                (self.class_invcovs is None and float(getattr(self.cfg, 'lcov_weight', 0.0)) > 0):
            fallback_means, fallback_invcovs = self._precompute_reference_semantic_stats()
            if self.feature_means is None:
                self.feature_means = fallback_means
            if self.class_invcovs is None:
                self.class_invcovs = fallback_invcovs
        self.global_step = 0

    def _load_synset(self) -> PyramidDataset:
        ckpt_path = self.task_dir / 'ckpt.pth'
        final_path = self.task_dir / 'final' / 'data.pth'
        root_path = self.task_dir / 'data.pth'
        if not ckpt_path.exists() and not final_path.exists() and not root_path.exists():
            raise FileNotFoundError(f'no checkpoint or distilled bundle found in {self.task_dir}')

        class DummyCfg:
            pass

        if ckpt_path.exists():
            load_dict = torch.load(ckpt_path, map_location='cpu')
            synset = load_dict['synset']
            first_level = synset['pyramid'][0]
            syn_res = int(first_level.shape[-1])
            dummy = DummyCfg()
            dummy.ipc = int(first_level.shape[0] // self.num_classes)
            dummy.lr = float(self.cfg.lr)
            dummy.init_mode = 'noise'
            dummy.syn_res = syn_res
            dummy.pyramid_start_res = syn_res
            dummy.decorrelate_color = True
            dummy.pyramid_extent_it = 10 ** 9
            dataset = PyramidDataset(num_classes=self.num_classes, cfg=dummy, device=self.device, out_dir=str(self.refresh_dir), init_synset_now=False)
            dataset.load_from_dict(synset)
            return dataset

        payload_path = final_path if final_path.exists() else root_path
        payload = torch.load(payload_path, map_location='cpu')
        images = payload['images'].float()
        labels = payload['labels'].long()
        syn_res = int(images.shape[-1])
        dummy = DummyCfg()
        dummy.ipc = int(images.shape[0] // self.num_classes)
        dummy.lr = float(self.cfg.lr)
        dummy.init_mode = 'noise'
        dummy.syn_res = syn_res
        dummy.pyramid_start_res = syn_res
        dummy.decorrelate_color = False
        dummy.pyramid_extent_it = 10 ** 9
        dataset = PyramidDataset(num_classes=self.num_classes, cfg=dummy, device=self.device, out_dir=str(self.refresh_dir), init_synset_now=False)
        dataset.load_from_images(images, labels)
        return dataset

    def _optimizer_params(self):
        params = []
        for group in self.distilled_dataset.optimizer.param_groups:
            params.extend(group.get('params', []))
        return params

    @staticmethod
    def _grads_are_finite(params) -> bool:
        for param in params:
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False
        return True

    def _cleanup_device_memory(self):
        gc.collect()
        if torch.cuda.is_available() and self.device.type == 'cuda':
            try:
                torch.cuda.synchronize(device=self.device)
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _reference_subset(self, idx: Tensor):
        idx_cpu = idx.detach().cpu()
        ref_images = self.reference_images[idx_cpu].to(self.device, non_blocking=True)
        ref_labels = self.reference_labels[idx_cpu].to(self.device, non_blocking=True)
        return ref_images, ref_labels

    def _iter_class_blocks(self):
        labels = self.reference_labels
        unique_classes = torch.unique(labels)
        if unique_classes.numel() == 0:
            return []
        perm = torch.randperm(unique_classes.numel())
        shuffled = unique_classes[perm]
        class_subsample = int(getattr(self.cfg, 'class_subsample', 0) or 0)
        if class_subsample > 0:
            shuffled = shuffled[:min(class_subsample, shuffled.numel())]
        chunk = max(1, int(getattr(self.cfg, 'class_chunk_size', 4) or 4))
        blocks = []
        for start in range(0, shuffled.numel(), chunk):
            picked = shuffled[start:start + chunk]
            mask = (labels[:, None] == picked[None, :]).any(dim=1)
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            if idx.numel() > 0:
                blocks.append(idx)
        return blocks

    def refresh(self):
        if self.new_feature_fn is None and self.new_token_fn is None:
            raise RuntimeError('new_feature_fn or new_token_fn must be set before calling refresh()')
        if self.refresh_dir.exists():
            shutil.rmtree(self.refresh_dir, ignore_errors=True)
        self.refresh_dir.mkdir(parents=True, exist_ok=True)
        stale_tmp = self.task_dir / 'tmp.pth'
        if stale_tmp.exists():
            try:
                stale_tmp.unlink()
            except OSError:
                pass
        self._cleanup_device_memory()
        for step in tqdm(range(1, self.cfg.iterations + 1), total=self.cfg.iterations, desc='Refreshing Synsets'):
            self.global_step = step
            self.step()
            self._cleanup_device_memory()

        with torch.no_grad():
            final_images, final_labels = self.distilled_dataset.get_data()
        # Keep only the final refreshed bundle in canonical locations used by replay.
        save_distilled_bundle(final_images.detach(), final_labels.detach(), self.task_dir, step=self.global_step, tag='refresh_canonical')
        save_distilled_bundle(final_images.detach(), final_labels.detach(), self.task_dir / 'final', step=self.global_step, tag='refresh_canonical')
        self._cleanup_device_memory()

    def _grad_from_features(self, feats: Tensor, labels: Tensor, fc: nn.Module, create_graph: bool) -> Tensor:
        with torch.enable_grad():
            # Teacher branch: no need to keep graph.
            # Student branch: keep feats connected to cur_images so that the
            # gradient-matching loss can update the distilled images.
            feats = feats if create_graph else feats.detach()
            labels = labels.detach().long()

            weight = fc.linear.weight.detach().clone().requires_grad_(True)
            bias = None
            if fc.linear.bias is not None:
                bias = fc.linear.bias.detach().clone().requires_grad_(True)

            out = F.linear(feats, weight, bias)
            loss = F.cross_entropy(out, labels)

            params = [weight] if bias is None else [weight, bias]
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=create_graph,
                create_graph=create_graph,
                allow_unused=False,
            )
            grad_w = grads[0]
            if bias is None:
                grad_b = torch.zeros(weight.size(0), device=grad_w.device, dtype=grad_w.dtype)
            else:
                grad_b = grads[1]
            return torch.cat([grad_w.flatten(), grad_b.flatten()], dim=0)

    def _angle_weighted_patch_loss(self, new_patch: Tensor, old_patch: Tensor, new_cls: Tensor) -> Tensor:
        if new_patch is None or old_patch is None or new_cls is None:
            return torch.tensor(0.0, device=self.device)
        new_patch_norm = F.normalize(new_patch, p=2, dim=-1)
        old_patch_norm = F.normalize(old_patch, p=2, dim=-1)
        alpha_cos = F.cosine_similarity(new_cls.unsqueeze(1), new_patch, dim=-1).clamp(min=-1.0, max=1.0)
        alpha_angle = 1 - (torch.acos(alpha_cos) / torch.pi)
        distances = torch.norm(new_patch_norm - old_patch_norm, p=2, dim=-1)
        weights = (1 - alpha_angle.detach())
        patch_sample_k = int(getattr(self.cfg, 'patch_sample_k', 0) or 0)
        if patch_sample_k > 0 and patch_sample_k < weights.size(1):
            top_idx = torch.topk(weights, k=patch_sample_k, dim=1, largest=True).indices
            distances = torch.gather(distances, 1, top_idx)
            weights = torch.gather(weights, 1, top_idx)
        return (weights * distances).mean()

    def _paired_augments(self, ref_images: Tensor, cur_images: Tensor):
        if getattr(self.cfg, 'shared_aug_refresh', True) and hasattr(self.syn_augmentor, 'paired'):
            return self.syn_augmentor.paired(ref_images, cur_images)
        return self.syn_augmentor(ref_images), self.syn_augmentor(cur_images)

    def _extract_teacher(self, ref_images: Tensor):
        old_cls = old_patch = old_feats = None
        if self.old_token_fn is not None:
            old_cls, old_patch = self.old_token_fn(ref_images)
            old_feats = old_cls
        elif self.old_feature_fn is not None:
            old_feats = self.old_feature_fn(ref_images)
        old_logits = self.old_logits_fn(ref_images) if self.old_logits_fn is not None else None
        return old_feats, old_cls, old_patch, old_logits

    def _extract_student(self, cur_images: Tensor):
        new_cls = new_patch = new_feats = None
        if self.new_token_fn is not None:
            new_cls, new_patch = self.new_token_fn(cur_images)
            new_feats = new_cls
        elif self.new_feature_fn is not None:
            new_feats = self.new_feature_fn(cur_images)
        new_logits = self.new_logits_fn(cur_images) if self.new_logits_fn is not None else None
        return new_feats, new_cls, new_patch, new_logits


    def _projected_target(self, ref_images: Tensor, old_feats: Tensor):
        proj = None
        if self.drift_projector_fn is not None and old_feats is not None:
            proj = self.drift_projector_fn(old_feats.detach())
        ref_new = None
        if self.new_token_fn is not None:
            with torch.no_grad():
                ref_new, _ = self.new_token_fn(ref_images)
        elif self.new_feature_fn is not None:
            with torch.no_grad():
                ref_new = self.new_feature_fn(ref_images)
        if proj is None:
            return ref_new
        if ref_new is None:
            return proj
        alpha = float(getattr(self.cfg, 'projector_mix_alpha', 0.7))
        return alpha * proj + (1.0 - alpha) * ref_new.detach()


    @staticmethod
    def _pairwise_mahalanobis(feats: Tensor, invcov: Tensor) -> Tensor:
        if feats.size(0) <= 1:
            return feats.new_zeros((0,))
        diff = feats.unsqueeze(1) - feats.unsqueeze(0)
        dist2 = torch.einsum('...d,df,...f->...', diff, invcov, diff).clamp_min(0.0)
        dist = torch.sqrt(dist2 + 1e-8)
        tri = torch.triu_indices(feats.size(0), feats.size(0), offset=1, device=feats.device)
        return dist[tri[0], tri[1]]

    def _precompute_reference_semantic_stats(self):
        if self.old_feature_fn is None and self.old_token_fn is None:
            means = torch.zeros(self.num_classes, self.num_feats)
            invcovs = torch.eye(self.num_feats).unsqueeze(0).repeat(self.num_classes, 1, 1)
            return means, invcovs
        loader_bs = max(1, int(getattr(self.cfg, 'precompute_batch_cap', 8) or 8))
        chunks = [[] for _ in range(self.num_classes)]
        for start in range(0, self.reference_images.size(0), loader_bs):
            end = min(start + loader_bs, self.reference_images.size(0))
            images = self.reference_images[start:end].to(self.device, non_blocking=True)
            labels = self.reference_labels[start:end].to(self.device, non_blocking=True).long()
            with torch.no_grad():
                feats, _, _, _ = self._extract_teacher(images)
            if feats is None:
                continue
            for c in labels.unique():
                c_int = int(c.item())
                if c_int < 0 or c_int >= self.num_classes:
                    continue
                mask = labels == c
                chunks[c_int].append(feats[mask].detach().cpu())
        means = []
        invcovs = []
        for c in range(self.num_classes):
            if chunks[c]:
                feats = torch.cat(chunks[c], dim=0).float().to(self.device)
                means.append(feats.mean(dim=0).detach().cpu())
                if feats.size(0) > 1:
                    cov = torch.cov(feats.T)
                else:
                    cov = torch.eye(self.num_feats, device=self.device)
                cov = cov + torch.eye(self.num_feats, device=self.device, dtype=cov.dtype) * 1e-3
                invcovs.append(torch.linalg.pinv(cov).detach().cpu())
            else:
                means.append(torch.zeros(self.num_feats))
                invcovs.append(torch.eye(self.num_feats))
        return torch.stack(means, dim=0), torch.stack(invcovs, dim=0)

    def _semantic_drift_alignment_loss(self, old_feats: Tensor, new_feats: Tensor, labels: Tensor):
        if new_feats is None or new_feats.numel() == 0 or labels.numel() == 0:
            zero = labels.new_zeros((), dtype=torch.float32).to(self.device)
            return zero, zero
        mean_loss = new_feats.new_zeros(())
        lcov_loss = new_feats.new_zeros(())
        mean_groups = 0
        cov_groups = 0
        labels = labels.detach().long()
        for c in labels.unique(sorted=True):
            idx = torch.nonzero(labels == c, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            c_int = int(c.item())
            if c_int < 0 or c_int >= self.num_classes:
                continue
            cls_new = new_feats.index_select(0, idx)
            if self.feature_means is not None:
                target_mean = self.feature_means[c_int].to(self.device, dtype=cls_new.dtype)
                mean_loss = mean_loss + F.mse_loss(cls_new.mean(dim=0), target_mean)
                mean_groups += 1
            if old_feats is not None and self.class_invcovs is not None and cls_new.size(0) > 1:
                cls_old = old_feats.index_select(0, idx).detach()
                invcov = self.class_invcovs[c_int].to(self.device, dtype=cls_new.dtype)
                new_dist = self._pairwise_mahalanobis(cls_new, invcov)
                old_dist = self._pairwise_mahalanobis(cls_old.to(self.device, dtype=cls_new.dtype), invcov).detach()
                if new_dist.numel() > 0 and old_dist.numel() > 0:
                    n = min(int(new_dist.numel()), int(old_dist.numel()))
                    lcov_loss = lcov_loss + F.l1_loss(new_dist[:n], old_dist[:n])
                    cov_groups += 1
        if mean_groups > 0:
            mean_loss = mean_loss / float(mean_groups)
        if cov_groups > 0:
            lcov_loss = lcov_loss / float(cov_groups)
        return mean_loss, lcov_loss

    def step(self):
        fc = _FC(num_feats=self.num_feats, num_classes=self.num_classes).to(self.device)
        class_blocks = self._iter_class_blocks()
        if not class_blocks:
            return float('nan')

        try:
            self.distilled_dataset.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.distilled_dataset.optimizer.zero_grad()

        total_loss_value = 0.0
        comp_raw = {k: 0.0 for k in ['gm','anchor','projector','feature_mean','lcov','patch','kd']}
        comp_weighted = {k: 0.0 for k in ['gm','anchor','projector','feature_mean','lcov','patch','kd']}
        valid_blocks = 0
        num_blocks = len(class_blocks)
        for subset_idx in class_blocks:
            ref_images, ref_labels = self._reference_subset(subset_idx)
            cur_images, labels = self.distilled_dataset.get_data_by_indices(subset_idx.to(self.device))
            anchor_loss = F.mse_loss(cur_images, ref_images)
            grad_loss_acc = cur_images.new_zeros(())
            drift_loss_acc = cur_images.new_zeros(())
            feature_mean_loss_acc = cur_images.new_zeros(())
            lcov_loss_acc = cur_images.new_zeros(())
            patch_loss_acc = cur_images.new_zeros(())
            kd_loss_acc = cur_images.new_zeros(())
            proj_loss_acc = cur_images.new_zeros(())

            for _ in range(max(1, int(self.cfg.augs_per_batch))):
                aug_ref, aug_cur = self._paired_augments(ref_images, cur_images)
                with torch.no_grad():
                    old_feats, old_cls, old_patch, old_logits = self._extract_teacher(aug_ref)
                old_grad = self._grad_from_features(old_feats.detach(), ref_labels, fc, create_graph=False).detach()
                new_feats, new_cls, new_patch, new_logits = self._extract_student(aug_cur)
                new_grad = self._grad_from_features(new_feats, labels, fc, create_graph=True)
                grad_loss_acc = grad_loss_acc + (1 - F.cosine_similarity(old_grad, new_grad, dim=0))
                proj_target = self._projected_target(aug_ref, old_feats)
                if self.cfg.projector_weight > 0 and proj_target is not None:
                    proj_loss_acc = proj_loss_acc + F.mse_loss(
                        F.normalize(new_feats, p=2, dim=1),
                        F.normalize(proj_target.detach(), p=2, dim=1),
                    )
                if self.cfg.feature_mean_weight > 0 or self.cfg.lcov_weight > 0:
                    mean_loss_i, lcov_loss_i = self._semantic_drift_alignment_loss(old_feats.detach(), new_feats, labels)
                    feature_mean_loss_acc = feature_mean_loss_acc + mean_loss_i
                    lcov_loss_acc = lcov_loss_acc + lcov_loss_i
                if self.cfg.patch_weight > 0 and old_patch is not None and new_patch is not None and new_cls is not None:
                    patch_loss_acc = patch_loss_acc + self._angle_weighted_patch_loss(new_patch, old_patch.detach(), new_cls)
                if self.cfg.kd_weight > 0 and old_logits is not None and new_logits is not None:
                    student_log_probs = F.log_softmax(new_logits / self.cfg.temperature, dim=1)
                    teacher_probs = F.softmax(old_logits.detach() / self.cfg.temperature, dim=1)
                    kd_loss_acc = kd_loss_acc + F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (self.cfg.temperature ** 2)

            aug_div = float(max(1, int(self.cfg.augs_per_batch)))
            gm_raw = grad_loss_acc / aug_div
            anchor_raw = anchor_loss
            projector_raw = proj_loss_acc / aug_div
            feature_mean_raw = feature_mean_loss_acc / aug_div
            lcov_raw = lcov_loss_acc / aug_div
            patch_raw = patch_loss_acc / aug_div
            kd_raw = kd_loss_acc / aug_div
            gm_w = self.cfg.grad_weight * gm_raw
            anchor_w = self.cfg.image_anchor_weight * anchor_raw
            projector_w = self.cfg.projector_weight * projector_raw
            feature_mean_w = self.cfg.feature_mean_weight * feature_mean_raw
            lcov_w = self.cfg.lcov_weight * lcov_raw
            patch_w = self.cfg.patch_weight * patch_raw
            kd_w = self.cfg.kd_weight * kd_raw
            block_loss = (gm_w + anchor_w + projector_w + feature_mean_w + lcov_w + patch_w + kd_w) / float(num_blocks)
            for _name, _raw, _w in [
                ('gm', gm_raw, gm_w),
                ('anchor', anchor_raw, anchor_w),
                ('projector', projector_raw, projector_w),
                ('feature_mean', feature_mean_raw, feature_mean_w),
                ('lcov', lcov_raw, lcov_w),
                ('patch', patch_raw, patch_w),
                ('kd', kd_raw, kd_w),
            ]:
                comp_raw[_name] += float(_raw.detach().item()) / float(num_blocks)
                comp_weighted[_name] += float(_w.detach().item()) / float(num_blocks)

            if not torch.isfinite(block_loss.detach()):
                warnings.warn(f'[Official-LGM-Refresh] Non-finite refresh loss at step {self.global_step}; skipping one block.')
                del ref_images, ref_labels, cur_images, labels, block_loss
                self._cleanup_device_memory()
                continue

            block_loss.backward()
            total_loss_value += float(block_loss.detach().item())
            valid_blocks += 1
            del ref_images, ref_labels, cur_images, labels, block_loss
            self._cleanup_device_memory()

        params = self._optimizer_params()
        if valid_blocks == 0:
            try:
                self.distilled_dataset.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.distilled_dataset.optimizer.zero_grad()
            return float('nan')
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(self.cfg.max_grad_norm))
        if not self._grads_are_finite(params):
            warnings.warn(f'[Official-LGM-Refresh] Non-finite synset gradients at step {self.global_step}; skipping optimizer step.')
            try:
                self.distilled_dataset.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.distilled_dataset.optimizer.zero_grad()
            return total_loss_value
        self.distilled_dataset.optimizer.step()
        self.last_loss_terms = {
            'total': float(total_loss_value),
            **{f'{k}': float(v) for k, v in comp_raw.items()},
            **{f'{k}_w': float(v) for k, v in comp_weighted.items()},
        }
        log_it = max(1, int(getattr(self.cfg, 'loss_log_it', getattr(self.cfg, 'image_log_it', 100)) or 100))
        if self.global_step % log_it == 0 or self.global_step == int(self.cfg.iterations):
            logging.info(
                '[Official-LGM-Refresh-Detail] Step %d/%d => Total %.4f | GM %.4f * %.4f = %.4f | Anchor %.4f * %.4f = %.4f | Projector %.4f * %.4f = %.4f | Mean %.4f * %.4f = %.4f | Lcov %.4f * %.4f = %.4f | Patch %.4f * %.4f = %.4f | KD %.4f * %.4f = %.4f',
                int(self.global_step), int(self.cfg.iterations), self.last_loss_terms['total'],
                self.last_loss_terms['gm'], float(self.cfg.grad_weight), self.last_loss_terms['gm_w'],
                self.last_loss_terms['anchor'], float(self.cfg.image_anchor_weight), self.last_loss_terms['anchor_w'],
                self.last_loss_terms['projector'], float(self.cfg.projector_weight), self.last_loss_terms['projector_w'],
                self.last_loss_terms['feature_mean'], float(self.cfg.feature_mean_weight), self.last_loss_terms['feature_mean_w'],
                self.last_loss_terms['lcov'], float(self.cfg.lcov_weight), self.last_loss_terms['lcov_w'],
                self.last_loss_terms['patch'], float(self.cfg.patch_weight), self.last_loss_terms['patch_w'],
                self.last_loss_terms['kd'], float(self.cfg.kd_weight), self.last_loss_terms['kd_w'],
            )
        return total_loss_value

    def save_checkpoint(self, to_root: bool = False):
        save_dict = {
            'synset': self.distilled_dataset.get_save_dict(),
            'global_step': self.global_step,
            'refresh_cfg': vars(self.cfg),
        }
        target_dir = self.task_dir if to_root else self.refresh_dir
        torch.save(save_dict, target_dir / 'ckpt.pth')
