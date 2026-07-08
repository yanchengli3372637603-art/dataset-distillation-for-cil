import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset

import logging
import gc
import numpy as np
from tqdm import tqdm

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from models.network import MANet
from models.attention import  Attention_HLoRA, Attention_LoRA, Attention_GLoRA

from utils.schedulers import CosineSchedule
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.toolkit import count_parameters
from models.losses import AngularPenaltySMLoss, MahalanobisLoss,compute_angle_weighted_patch_distillation_loss
import re
from pathlib import Path

from utils.distilled_replay import DistilledReplayDataset, load_distilled_replay
from lgm_official.utils import temporarily_freeze_model, temporarily_set_grad_checkpointing


class Learner(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = MANet(args)
        for module in self._network.modules():
            if isinstance(module, Attention_HLoRA):
                module.init_param()
            if isinstance(module, Attention_LoRA):
                module.init_param()
            if isinstance(module, Attention_GLoRA):
                module.init_param()
            
        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.scale = args["scale"]
        self.margin = args["margin"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]
        self.logit_norm = 0.1
        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.task_sizes = []

        # class prototypes
        self._class_means = None
        self._class_covs = None
        self._old_class_covs = None
        self.acc_matrix = np.zeros((self.total_sessions, self.total_sessions))
        self.replay_loader = None
        self._pending_lgm_dataset = None
        self._pending_lgm_label_offset = None
        self._pending_lgm_task_size = None
        self._pending_lgm_test_dataset = None
        self.distilled_probe_curve = []

    def after_task(self):
        self._old_network = self._network.copy().freeze().cpu()
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self._old_class_covs = None

    def incremental_train(self, data_manager):

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=True)
        self.replay_loader = self._build_distilled_replay_loader(data_manager)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      num_workers=self.num_workers, pin_memory=True)
        
        # Semantic Shift old embedding 
        if self._cur_task > 0 and self._old_network is not None:
            self._old_network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._old_network, self._cur_task-1)
            need_lgm_lcov = (self._should_run_official_lgm() or self._should_refresh_old_synsets())
            if self.args['cc'] is True or need_lgm_lcov:
                # Reuse the MACIL covariance path for both training-time CC and
                # post-eval distilled-image semantic-drift constraints.
                self._old_class_covs = self._compute_class_invcov(data_manager)
            self._old_network.cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._train(self.train_loader, self.test_loader)
        self.replay_loader = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Semantic Shift
        if self._cur_task > 0:
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network)
            old_class_mean = self._class_means[:self._known_classes]
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 1.0)
            if self.args['msc'] is True:
                old_class_mean += gap
                self._class_means[:self._known_classes] = old_class_mean


        # update mean and cov and classifier alignment
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task > 0 and self.args['ca'] is True:
            self._stage2_compact_classifier(self.task_sizes[-1])

        if self._should_run_official_lgm():
            # Distillation still uses train-set images, but adaptive class boosting is
            # judged on the full current-task test split to match the paper-style
            # synthetic-set linear-probe evaluation more closely.
            self._pending_lgm_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source='train', mode='test'
            )
            self._pending_lgm_label_offset = self._known_classes
            self._pending_lgm_task_size = self.task_sizes[-1]
            self._pending_lgm_test_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source='test', mode='test'
            )
        else:
            self._pending_lgm_dataset = None
            self._pending_lgm_label_offset = None
            self._pending_lgm_task_size = None
            self._pending_lgm_test_dataset = None


    def _should_run_official_lgm(self):
        return bool(self.args.get('enable_official_lgm', False))

    def _should_refresh_old_synsets(self):
        return bool(self.args.get('enable_official_lgm_refresh', False)) and self._cur_task > 0 and self._old_network is not None

    def run_post_eval_hooks(self):
        if self._should_refresh_old_synsets():
            self._refresh_old_task_synsets()
        if self._should_run_official_lgm() and self._pending_lgm_dataset is not None:
            self._distill_current_task_with_official_lgm(
                distill_dataset=self._pending_lgm_dataset,
                label_offset=int(self._pending_lgm_label_offset),
                task_size=int(self._pending_lgm_task_size),
                test_dataset=self._pending_lgm_test_dataset,
            )
            self._pending_lgm_dataset = None
            self._pending_lgm_label_offset = None
            self._pending_lgm_task_size = None
            self._pending_lgm_test_dataset = None
        if self._should_run_distilled_linear_eval():
            self._run_distilled_linear_eval()

    def _should_use_distilled_replay(self):
        return bool(self.args.get('enable_distilled_replay', False)) and self._cur_task > 0 and self._known_classes > 0

    def _build_distilled_replay_loader(self, data_manager):
        if not self._should_use_distilled_replay():
            return None

        seed = self.args.get('seed', 'unknown')
        replay_root = Path(self.args['logdir']) / 'lgm_official' / f'seed_{seed}'
        replay_images, replay_labels, replay_task_ids, used_paths = load_distilled_replay(
            base_dir=replay_root,
            upto_task_exclusive=self._cur_task,
            task_sizes=self.task_sizes,
            return_task_ids=True,
        )
        if replay_images is None or replay_labels is None or len(replay_labels) == 0:
            logging.info('[Distilled-Replay] No replay bundle found. Training will use current-task real data only.')
            return None

        train_trsf = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train', mode='train').trsf
        replay_dataset = DistilledReplayDataset(replay_images, replay_labels, trsf=train_trsf, task_ids=replay_task_ids)
        replay_batch_size = int(self.args.get('distilled_replay_batch_size', self.batch_size))
        sampler = None
        shuffle = True
        if bool(self.args.get('distilled_replay_class_balanced', True)):
            unique_labels, counts = torch.unique(replay_labels, return_counts=True)
            inv_freq = {int(lbl.item()): 1.0 / float(cnt.item()) for lbl, cnt in zip(unique_labels, counts)}
            sample_weights = torch.tensor([inv_freq[int(lbl.item())] for lbl in replay_labels], dtype=torch.double)
            current_task_size = int(self.task_sizes[-1]) if len(self.task_sizes) > 0 else max(1, self.class_num)
            default_repeat = max(1, int(np.ceil(max(1, self._known_classes) / max(1, current_task_size))))
            replay_repeat = int(self.args.get('distilled_replay_repeat_factor', default_repeat))
            num_samples = int(len(replay_dataset) * max(1, replay_repeat))
            sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
            shuffle = False
            logging.info(f'[Distilled-Replay] Using class-balanced replay sampler with repeat_factor={max(1, replay_repeat)} and num_samples={num_samples}.')

        replay_loader = DataLoader(
            replay_dataset,
            batch_size=max(1, replay_batch_size),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=min(self.num_workers, 2),
            pin_memory=True,
            drop_last=False,
        )
        logging.info(f'[Distilled-Replay] Loaded {len(replay_dataset)} synthetic old-class samples from {len(used_paths)} task bundles.')
        for path in used_paths:
            logging.info(f'[Distilled-Replay]   using: {path}')
        return replay_loader

    def _build_lgm_config(self):
        from lgm_official import LGMConfig

        return LGMConfig(
            ipc=int(self.args.get('distill_ipc', 2)),
            lr=float(self.args.get('distill_lr', 2e-3)),
            iterations=int(self.args.get('distill_iterations', 5000)),
            augs_per_batch=int(self.args.get('distill_augs_per_batch', 10)),
            distill_mode='pyramid',
            aug_mode=self.args.get('distill_aug_mode', 'standard'),
            decorrelate_color=bool(self.args.get('distill_decorrelate_color', True)),
            init_mode=self.args.get('distill_init_mode', 'noise'),
            pyramid_extent_it=int(self.args.get('distill_pyramid_extent_it', 200)),
            pyramid_start_res=int(self.args.get('distill_pyramid_start_res', 1)),
            image_log_it=int(self.args.get('distill_image_log_it', 500)),
            loss_log_it=int(self.args.get('distill_loss_log_it', 500)),
            checkpoint_it=int(self.args.get('distill_checkpoint_it', 100)),
            syn_res=int(self.args.get('distill_syn_res', 224)),
            crop_res=int(self.args.get('distill_crop_res', 224)),
            noise_std=float(self.args.get('distill_noise_std', 0.0)),
            save_every=int(self.args.get('distill_save_every', 200)),
            resume=bool(self.args.get('distill_resume', True)),
            real_batch_cap=int(self.args.get('distill_real_batch_cap', 64)),
            syn_batch_cap=int(self.args.get('distill_syn_batch_cap', 40)),
            max_grad_norm=float(self.args.get('distill_max_grad_norm', 1.0)),
            feature_mean_weight=float(self.args.get('distill_feature_mean_weight', 0.0)),
            lcov_weight=float(self.args.get('distill_lcov_weight', 0.0)),
            patch_weight=float(self.args.get('distill_patch_weight', 0.0)),
            patch_sample_k=int(self.args.get('distill_patch_sample_k', 0)),
            stat_batch_size=int(self.args.get('distill_stat_batch_size', self.args.get('distill_real_batch_cap', 64))),
            diversity_weight=float(self.args.get('distill_diversity_weight', 0.01)),
            diversity_margin=float(self.args.get('distill_diversity_margin', 0.8)),
            match_ipc=int(self.args.get('distill_match_ipc', 1)),
            adaptive_enable=bool(self.args.get('distill_adaptive_enable', False)),
            adaptive_rounds=int(self.args.get('distill_adaptive_rounds', 0)),
            adaptive_eval_max_per_class=int(self.args.get('distill_adaptive_eval_max_per_class', 64)),
            adaptive_eval_use_testset=bool(self.args.get('distill_adaptive_eval_use_testset', True)),
            adaptive_eval_batch_size=int(self.args.get('distill_adaptive_eval_batch_size', 100)),
            adaptive_eval_epochs=int(self.args.get('distill_adaptive_eval_epochs', 1000)),
            adaptive_eval_lr=float(self.args.get('distill_adaptive_eval_lr', 1e-3)),
            adaptive_eval_weight_decay=float(self.args.get('distill_adaptive_eval_weight_decay', 0.0)),
            adaptive_eval_patience=int(self.args.get('distill_adaptive_eval_patience', 50)),
            adaptive_eval_log_interval=int(self.args.get('distill_adaptive_eval_log_interval', 50)),
            adaptive_ridge=float(self.args.get('distill_adaptive_ridge', 1e-3)),
            adaptive_drop_margin=float(self.args.get('distill_adaptive_drop_margin', 2.0)),
            adaptive_low_margin=float(self.args.get('distill_adaptive_low_margin', 5.0)),
            adaptive_min_class_acc=float(self.args.get('distill_adaptive_min_class_acc', 70.0)),
            adaptive_extra_ipc=int(self.args.get('distill_adaptive_extra_ipc', 1)),
            adaptive_max_ipc=int(self.args.get('distill_adaptive_max_ipc', max(3, int(self.args.get('distill_ipc', 2)) + 1))),
            adaptive_extra_iterations=int(self.args.get('distill_adaptive_extra_iterations', 600)),
            adaptive_focus_only=bool(self.args.get('distill_adaptive_focus_only', True)),
            adaptive_save_best=bool(self.args.get('distill_adaptive_save_best', True)),
        )


    def _collect_distill_semantic_targets(self, dataset, model, device, task_id, label_offset: int, num_classes: int):
        """Estimate class feature means and inverse covariances for distilled-image losses.

        Covariance estimation intentionally goes through ``self.shrink_cov`` so the
        post-eval distillation path uses the same covariance shrinkage as MACIL's
        training-time ``_compute_class_invcov`` implementation.
        """
        if dataset is None or len(dataset) == 0 or num_classes <= 0:
            return None, None, None
        batch_size = max(1, int(self.args.get('distill_stat_batch_size', self.args.get('distill_real_batch_cap', 64))))
        max_per_class = int(self.args.get('distill_lcov_max_features_per_class', 256))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(self.num_workers, 2),
            pin_memory=True,
            drop_last=False,
        )
        chunks = [[] for _ in range(num_classes)]
        counts = [0 for _ in range(num_classes)]
        model.eval()
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).long() - int(label_offset)
                feats = model.extract_vector(inputs, task_id=task_id).detach()
                valid = (targets >= 0) & (targets < num_classes)
                if not valid.any():
                    continue
                feats = feats[valid]
                targets = targets[valid]
                for cls_tensor in targets.unique(sorted=True):
                    cls_id = int(cls_tensor.item())
                    mask = targets == cls_tensor
                    cls_feats = feats[mask].detach().cpu().float()
                    if max_per_class > 0:
                        remaining = max(0, max_per_class - counts[cls_id])
                        if remaining <= 0:
                            continue
                        cls_feats = cls_feats[:remaining]
                    chunks[cls_id].append(cls_feats)
                    counts[cls_id] += int(cls_feats.size(0))

        means = torch.zeros((num_classes, self.feature_dim), dtype=torch.float32)
        invcovs = torch.zeros((num_classes, self.feature_dim, self.feature_dim), dtype=torch.float32)
        ref_features = []
        eye = torch.eye(self.feature_dim, device=device, dtype=torch.float64)
        for cls_id in range(num_classes):
            if chunks[cls_id]:
                feats_cpu = torch.cat(chunks[cls_id], dim=0).float()
                ref_features.append(feats_cpu.detach().cpu())
                feats_dev = feats_cpu.to(device=device, dtype=torch.float64)
                means[cls_id] = feats_cpu.mean(dim=0)
                if feats_dev.size(0) > 1:
                    cov = torch.cov(feats_dev.T)
                else:
                    cov = eye.clone()
                cov = self.shrink_cov(cov) + eye * 1e-3
                invcovs[cls_id] = torch.linalg.pinv(cov).detach().cpu().float()
            else:
                ref_features.append(None)
                invcovs[cls_id] = torch.eye(self.feature_dim, dtype=torch.float32)
        return means, invcovs, ref_features

    def _stored_class_invcovs(self, class_offset: int, task_size: int, device):
        if self._class_covs is None or task_size <= 0:
            return None
        if class_offset + task_size > self._class_covs.size(0):
            return None
        invcovs = []
        eye = torch.eye(self.feature_dim, device=device, dtype=torch.float64)
        for class_idx in range(class_offset, class_offset + task_size):
            cov = self._class_covs[class_idx].to(device=device, dtype=torch.float64)
            cov = self.shrink_cov(cov) + eye * 1e-3
            invcovs.append(torch.linalg.pinv(cov).detach().cpu().float())
        return torch.stack(invcovs, dim=0) if invcovs else None



    def _clear_cuda_residuals(self):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _clear_cuda_residuals_on(self, device=None):
        gc.collect()
        if not torch.cuda.is_available():
            return
        if device is None or (isinstance(device, torch.device) and device.type == 'cuda'):
            try:
                if device is not None:
                    torch.cuda.synchronize(device)
                else:
                    torch.cuda.synchronize()
            except Exception:
                pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def _resolve_posteval_device(self):
        if not torch.cuda.is_available():
            return self._device
        use_dual = bool(self.args.get('enable_dual_gpu_posteval', True))
        if not use_dual:
            return self._device
        preferred = self.args.get('posteval_device', 'auto')
        primary_index = self._device.index if isinstance(self._device, torch.device) and self._device.type == 'cuda' else None
        if isinstance(preferred, torch.device):
            return preferred
        if preferred not in (None, '', 'auto'):
            preferred_str = str(preferred).strip().replace('，', ',')
            if preferred_str.startswith('cuda:'):
                try:
                    idx = int(preferred_str.split(':', 1)[1])
                    if primary_index is None or idx != primary_index:
                        return torch.device(f'cuda:{idx}')
                except Exception:
                    pass
            else:
                try:
                    idx = int(preferred_str)
                    if primary_index is None or idx != primary_index:
                        return torch.device(f'cuda:{idx}')
                except Exception:
                    pass
        candidates = []
        for idx in range(torch.cuda.device_count()):
            if primary_index is not None and idx == primary_index:
                continue
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            except Exception:
                free_bytes, total_bytes = (0, 0)
            candidates.append((free_bytes, total_bytes, idx))
        if not candidates:
            return self._device
        candidates.sort(reverse=True)
        return torch.device(f'cuda:{candidates[0][2]}')

    def _build_refresh_config(self):
        from lgm_official import RefreshConfig

        return RefreshConfig(
            lr=float(self.args.get('refresh_lr', self.args.get('distill_lr', 1e-3))),
            iterations=int(self.args.get('refresh_iterations', 300)),
            augs_per_batch=int(self.args.get('refresh_augs_per_batch', 1)),
            aug_mode=self.args.get('refresh_aug_mode', self.args.get('distill_aug_mode', 'standard')),
            image_log_it=int(self.args.get('refresh_image_log_it', 100)),
            loss_log_it=int(self.args.get('refresh_loss_log_it', 100)),
            checkpoint_it=int(self.args.get('refresh_checkpoint_it', 100)),
            save_every=int(self.args.get('refresh_save_every', 100)),
            crop_res=int(self.args.get('refresh_crop_res', self.args.get('distill_crop_res', 224))),
            noise_std=float(self.args.get('refresh_noise_std', self.args.get('distill_noise_std', 0.0))),
            syn_batch_cap=int(self.args.get('refresh_syn_batch_cap', self.args.get('distill_syn_batch_cap', 2))),
            grad_weight=float(self.args.get('refresh_grad_weight', 1.0)),
            image_anchor_weight=float(self.args.get('refresh_image_anchor_weight', 0.05)),
            kd_weight=float(self.args.get('refresh_kd_weight', 0.1)),
            temperature=float(self.args.get('refresh_temperature', 2.0)),
            max_grad_norm=float(self.args.get('refresh_max_grad_norm', 1.0)),
            precompute_batch_cap=int(self.args.get('refresh_precompute_batch_cap', 8)),
            class_subsample=int(self.args.get('refresh_class_subsample', 0)),
            class_chunk_size=int(self.args.get('refresh_class_chunk_size', 4)),
            feature_mean_weight=float(self.args.get('refresh_feature_mean_weight', 0.0)),
            lcov_weight=float(self.args.get('refresh_lcov_weight', 0.0)),
            patch_weight=float(self.args.get('refresh_patch_weight', 0.05)),
            patch_sample_k=int(self.args.get('refresh_patch_sample_k', 0)),
            shared_aug_refresh=bool(self.args.get('refresh_shared_aug', True)),
            projector_weight=float(self.args.get('refresh_projector_weight', 0.5)),
            projector_mix_alpha=float(self.args.get('refresh_projector_mix_alpha', 0.7)),
        )

    def _fit_refresh_projector(self, dataset, old_feature_fn, new_feature_fn, device, max_samples=1024, batch_size=64, ridge=1e-3):
        if dataset is None or old_feature_fn is None or new_feature_fn is None:
            return None
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(batch_size)),
            shuffle=False,
            num_workers=min(self.num_workers, 2),
            pin_memory=True,
            drop_last=False,
        )
        old_chunks, new_chunks = [], []
        seen = 0
        with torch.no_grad():
            for _, inputs, _ in loader:
                inputs = inputs.to(device, non_blocking=True)
                old_feats = old_feature_fn(inputs).detach()
                new_feats = new_feature_fn(inputs).detach()
                keep = old_feats.size(0)
                if max_samples > 0:
                    keep = min(keep, max(0, int(max_samples) - seen))
                if keep <= 0:
                    break
                old_chunks.append(old_feats[:keep])
                new_chunks.append(new_feats[:keep])
                seen += int(keep)
                if max_samples > 0 and seen >= int(max_samples):
                    break
        if not old_chunks:
            return None
        X = torch.cat(old_chunks, dim=0).float()
        Y = torch.cat(new_chunks, dim=0).float()
        if X.size(0) < 2:
            return None
        ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
        Xb = torch.cat([X, ones], dim=1)
        dim = Xb.size(1)
        reg = float(ridge) * torch.eye(dim, device=X.device, dtype=X.dtype)
        reg[-1, -1] = 0.0
        lhs = Xb.t().matmul(Xb) + reg
        rhs = Xb.t().matmul(Y)
        try:
            solution = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            solution = torch.linalg.pinv(lhs).matmul(rhs)
        W = solution[:-1].detach()
        b = solution[-1].detach()
        fit_mse = float(((Xb.matmul(solution) - Y) ** 2).mean().item())
        logging.info(f'[Official-LGM-Refresh] Fitted drift projector on {X.size(0)} samples with mse={fit_mse:.6f}')

        def projector_fn(z):
            return z.matmul(W.to(z.device, dtype=z.dtype)) + b.to(z.device, dtype=z.dtype)

        return projector_fn

    def _refresh_old_task_synsets(self):
        cfg = self._build_refresh_config()
        seed = self.args.get('seed', 'unknown')
        base_dir = Path(self.args['logdir']) / 'lgm_official' / f'seed_{seed}'
        new_backbone = self._network
        old_backbone = self._old_network
        refresh_on_cpu = bool(self.args.get('refresh_force_cpu', False))
        refresh_device = torch.device('cpu') if refresh_on_cpu else self._resolve_posteval_device()
        logging.info(f'[Official-LGM-Refresh] Using post-eval refresh device: {refresh_device}')
        self._clear_cuda_residuals()
        if refresh_device != self._device:
            self._clear_cuda_residuals_on(refresh_device)

        refresher = None
        try:
            old_backbone.to(refresh_device)
            new_backbone.to(refresh_device)
            for task_id in range(self._cur_task):
                task_size = int(self.task_sizes[task_id])
                class_offset = int(sum(self.task_sizes[:task_id]))
                task_dir = base_dir / f'task_{task_id:02d}'

                from lgm_official import SynsetRefreshForMACIL

                logging.info(
                    f'[Official-LGM-Refresh] Start refreshing task {task_id} synset using model '
                    f'{self._cur_task - 1} -> {self._cur_task} at {task_dir} '
                    f'(device={refresh_device})'
                )
                try:
                    def old_feature_fn(x, _task_id=task_id):
                        return old_backbone.extract_vector(x, task_id=_task_id)

                    def old_token_fn(x, _task_id=task_id):
                        return old_backbone.extract_tokens(x, task_id=_task_id)

                    def old_logits_fn(x, _offset=class_offset, _size=task_size):
                        return old_backbone.interface(x)[:, _offset:_offset + _size]

                    def new_feature_fn(x, _task_id=task_id):
                        return new_backbone.extract_vector(x, task_id=_task_id)

                    def new_token_fn(x, _task_id=task_id):
                        return new_backbone.extract_tokens(x, task_id=_task_id)

                    def new_logits_fn(x, _offset=class_offset, _size=task_size):
                        return new_backbone.interface(x)[:, _offset:_offset + _size]

                    projector_fn = self._fit_refresh_projector(
                        dataset=self._pending_lgm_dataset,
                        old_feature_fn=old_feature_fn,
                        new_feature_fn=new_feature_fn,
                        device=refresh_device,
                        max_samples=int(self.args.get('refresh_projector_max_samples', 1024)),
                        batch_size=int(self.args.get('refresh_projector_batch_size', 64)),
                        ridge=float(self.args.get('refresh_projector_ridge', 1e-3)),
                    )
                    refresh_feature_means = None
                    if self._class_means is not None and class_offset + task_size <= self._class_means.size(0):
                        refresh_feature_means = self._class_means[class_offset:class_offset + task_size].detach().cpu().float()
                    refresh_invcovs = self._stored_class_invcovs(class_offset, task_size, refresh_device)

                    with temporarily_freeze_model(old_backbone), temporarily_set_grad_checkpointing(old_backbone, enable=False),                          temporarily_freeze_model(new_backbone), temporarily_set_grad_checkpointing(new_backbone, enable=True):
                        refresher = SynsetRefreshForMACIL(
                            cfg=cfg,
                            old_feature_fn=old_feature_fn,
                            new_feature_fn=new_feature_fn,
                            num_feats=new_backbone.feature_dim,
                            num_classes=task_size,
                            device=refresh_device,
                            task_dir=str(task_dir),
                            refresh_tag=f'to_task_{self._cur_task:02d}',
                            old_logits_fn=old_logits_fn,
                            new_logits_fn=new_logits_fn,
                            old_token_fn=old_token_fn,
                            new_token_fn=new_token_fn,
                            drift_projector_fn=projector_fn,
                            feature_means=refresh_feature_means,
                            class_invcovs=refresh_invcovs,
                        )
                        refresher.refresh()
                    logging.info(f'[Official-LGM-Refresh] Finished refreshing task {task_id} synset at {task_dir}')
                except FileNotFoundError as exc:
                    logging.info(f'[Official-LGM-Refresh] Skip task {task_id}: {exc}')
                finally:
                    if refresher is not None:
                        del refresher
                        refresher = None
                    self._clear_cuda_residuals()
        finally:
            old_backbone.cpu()
            new_backbone.to(self._device)
            self._clear_cuda_residuals()
            if self._old_network is not None:
                self._old_network.cpu()

    def _distill_current_task_with_official_lgm(self, distill_dataset, label_offset: int, task_size: int, test_dataset=None):
        if task_size <= 0 or len(distill_dataset) == 0:
            return
        cfg = self._build_lgm_config()
        seed = self.args.get('seed', 'unknown')
        base_dir = Path(self.args['logdir']) / 'lgm_official' / f'seed_{seed}' / f'task_{self._cur_task:02d}'
        base_dir.mkdir(parents=True, exist_ok=True)

        # Reuse the current network in-place instead of cloning a second copy on GPU.
        # Cloning the backbone roughly doubles model memory and is the main cause of
        # OOM on 24 GB cards during paper-like LGM runs.
        distill_device = self._resolve_posteval_device()
        backbone = self._network.to(distill_device)
        task_id = backbone.numtask - 1
        logging.info(f'[Official-LGM] Using post-eval distill device: {distill_device}')
        self._clear_cuda_residuals()
        if distill_device != self._device:
            self._clear_cuda_residuals_on(distill_device)

        feature_means, class_invcovs, reference_features_by_class = self._collect_distill_semantic_targets(
            dataset=distill_dataset,
            model=backbone,
            device=distill_device,
            task_id=task_id,
            label_offset=label_offset,
            num_classes=task_size,
        )
        if self._old_class_covs is not None and self._old_class_covs.size(0) >= task_size:
            # ``_old_class_covs`` stores inverse covariances produced by MACIL's
            # _compute_class_invcov(data_manager), so reuse it directly for Lcov.
            class_invcovs = self._old_class_covs[:task_size].detach().cpu().float()

        with temporarily_freeze_model(backbone), temporarily_set_grad_checkpointing(backbone, enable=True):
            def feature_fn(x):
                feats = backbone.extract_vector(x, task_id=task_id)
                return feats

            def token_fn(x):
                cls_tokens, patch_tokens = backbone.extract_tokens(x, task_id=task_id)
                return cls_tokens, patch_tokens

            from lgm_official import LinearGMForMACIL

            distiller = LinearGMForMACIL(
                cfg=cfg,
                feature_fn=feature_fn,
                token_fn=token_fn,
                num_feats=backbone.feature_dim,
                num_classes=task_size,
                train_dataset=distill_dataset,
                device=distill_device,
                log_dir=str(base_dir),
                label_offset=label_offset,
                test_dataset=test_dataset,
                feature_means=feature_means,
                class_invcovs=class_invcovs,
                reference_features_by_class=reference_features_by_class,
            )
            logging.info(
                f'[Official-LGM] Start distillation for task {self._cur_task} at {base_dir} '
                f'(task_size={task_size}, ipc={cfg.ipc}, match_ipc={cfg.match_ipc}, syn_images={task_size * int(cfg.ipc)}, div_w={cfg.diversity_weight}, mean_w={cfg.feature_mean_weight}, lcov_w={cfg.lcov_weight}, adaptive={cfg.adaptive_enable}, adaptive_rounds={cfg.adaptive_rounds})'
            )
            try:
                distiller.distill()
            finally:
                del distiller
                gc.collect()
                if distill_device != self._device:
                    backbone.to(self._device)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        logging.info(f'[Official-LGM] Finished distillation for task {self._cur_task}; outputs saved to {base_dir}')

    def _should_run_distilled_linear_eval(self):
        return bool(self.args.get('enable_official_lgm_linear_eval', False))

    def _extract_feature_tensor(self, images: torch.Tensor, model, device: torch.device, batch_size: int = 64, task_id=None):
        feats = []
        model.eval()
        for start in range(0, images.size(0), batch_size):
            end = min(start + batch_size, images.size(0))
            batch = images[start:end].to(device, non_blocking=True)
            with torch.no_grad():
                feat = model.extract_vector(batch, task_id=task_id)
            feats.append(feat.detach().cpu())
        return torch.cat(feats, dim=0) if feats else torch.empty((0, self.feature_dim))

    def _collect_test_feature_tensor(self, model, device: torch.device, batch_size: int = 64, task_id=None):
        dataset = self.test_loader.dataset
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=min(self.num_workers, 2), pin_memory=True)
        feats, labels = [], []
        model.eval()
        for _, inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            with torch.no_grad():
                feat = model.extract_vector(inputs, task_id=task_id)
            feats.append(feat.detach().cpu())
            labels.append(targets.detach().cpu())
        if not feats:
            return torch.empty((0, self.feature_dim)), torch.empty((0,), dtype=torch.long)
        return torch.cat(feats, dim=0), torch.cat(labels, dim=0).long()

    def _run_distilled_linear_eval(self):
        if not self._should_run_distilled_linear_eval():
            return
        seed = self.args.get('seed', 'unknown')
        replay_root = Path(self.args['logdir']) / 'lgm_official' / f'seed_{seed}'
        syn_images, syn_labels, used_paths = load_distilled_replay(
            base_dir=replay_root,
            upto_task_exclusive=self._cur_task + 1,
            task_sizes=self.task_sizes,
        )
        if syn_images is None or syn_labels is None or syn_labels.numel() == 0:
            logging.info('[Official-LGM-Eval] Skip linear-eval: no distilled bundles available yet.')
            return

        eval_device = self._resolve_posteval_device()
        eval_bs = int(self.args.get('lgm_linear_eval_batch_size', 64))
        eval_epochs = int(self.args.get('lgm_linear_eval_epochs', 200))
        eval_lr = float(self.args.get('lgm_linear_eval_lr', 0.05))
        eval_wd = float(self.args.get('lgm_linear_eval_weight_decay', 0.0))
        eval_log_it = max(1, int(self.args.get('lgm_linear_eval_log_interval', 50)))
        task_id = self._cur_task

        backbone = self._network.to(eval_device)
        try:
            with temporarily_freeze_model(backbone), temporarily_set_grad_checkpointing(backbone, enable=False):
                syn_features = self._extract_feature_tensor(syn_images, backbone, eval_device, batch_size=eval_bs, task_id=task_id)
                test_features, test_labels = self._collect_test_feature_tensor(backbone, eval_device, batch_size=eval_bs, task_id=task_id)

            syn_dataset = TensorDataset(syn_features.float(), syn_labels.long())
            syn_loader = DataLoader(syn_dataset, batch_size=max(1, eval_bs), shuffle=True, num_workers=0, drop_last=False)

            classifier = torch.nn.Linear(self.feature_dim, self._total_classes, bias=True).to(eval_device)
            optimizer = optim.SGD(classifier.parameters(), lr=eval_lr, momentum=0.9, weight_decay=eval_wd)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max(1, eval_epochs))

            best_acc = 0.0
            final_acc = 0.0
            for epoch in range(eval_epochs):
                classifier.train()
                epoch_loss = 0.0
                seen = 0
                for batch_feats, batch_labels in syn_loader:
                    batch_feats = batch_feats.to(eval_device, non_blocking=True)
                    batch_labels = batch_labels.to(eval_device, non_blocking=True)
                    logits = classifier(batch_feats)
                    loss = F.cross_entropy(logits, batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.detach().item()) * batch_labels.size(0)
                    seen += int(batch_labels.size(0))
                scheduler.step()

                classifier.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for start in range(0, test_features.size(0), eval_bs):
                        end = min(start + eval_bs, test_features.size(0))
                        feats = test_features[start:end].to(eval_device, non_blocking=True)
                        labels = test_labels[start:end].to(eval_device, non_blocking=True)
                        preds = classifier(feats).argmax(dim=1)
                        correct += int((preds == labels).sum().item())
                        total += int(labels.size(0))
                final_acc = 100.0 * correct / max(1, total)
                best_acc = max(best_acc, final_acc)
                if (epoch + 1) % eval_log_it == 0 or epoch == 0 or epoch + 1 == eval_epochs:
                    logging.info(
                        f'[Official-LGM-Eval] Task {self._cur_task} Epoch {epoch + 1}/{eval_epochs} => '
                        f'Loss {epoch_loss / max(1, seen):.4f}, Test_accy {final_acc:.2f}, Best {best_acc:.2f}'
                    )

            self.distilled_probe_curve.append(float(final_acc))
            logging.info(f'[Official-LGM-Eval] Task {self._cur_task} distilled-only linear probe final acc: {final_acc:.2f}; best acc: {best_acc:.2f}')
            for path in used_paths:
                logging.info(f'[Official-LGM-Eval]   using distilled bundle: {path}')
        finally:
            self._network.to(self._device)
            self._clear_cuda_residuals()

    @staticmethod
    def _logit_kd_loss(student_logits, teacher_logits, temperature=2.0):
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task > 0 and self._old_network is not None:
            self._old_network.to(self._device)

        adapter_params = []
        current_head_params = []
        old_head_params = []
        replay_active = self.replay_loader is not None and self._cur_task > 0

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)
            classifier_match = re.search(r"(^|\.)classifier_pool\.(\d+)($|\.)", name)
            if classifier_match is not None:
                classifier_idx = int(classifier_match.group(2))
                if classifier_idx == self._network.numtask - 1:
                    param.requires_grad_(True)
                    current_head_params.append(param)
                    continue
                if replay_active and classifier_idx < self._network.numtask - 1:
                    param.requires_grad_(True)
                    old_head_params.append(param)
                    continue
            if self.args['lora_type'] == 'elora':
                if re.search(rf"(^|\.)lora_B_k\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue
                if re.search(rf"(^|\.)lora_B_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue
                if re.search(rf"(^|\.)lora_A_k\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue
                if re.search(rf"(^|\.)lora_A_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue
            if self.args['lora_type'] == 'hlora' or self.args['lora_type'] == 'glora':
                if re.search(rf"(^|\.)elora_B_k\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue
                if re.search(rf"(^|\.)elora_B_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue
                if re.search(rf"(^|\.)glora_B_k($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue
                if re.search(rf"(^|\.)glora_B_v($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue
                if re.search(rf"(^|\.)glora_A_k($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue
                if re.search(rf"(^|\.)glora_A_v($|\.)", name) is not None:
                    param.requires_grad_(True)
                    adapter_params.append(param)
                    continue

        old_head_lr_scale = float(self.args.get('distilled_replay_old_head_lr_scale', 0.25))
        network_params = []
        base_params = adapter_params + current_head_params
        if base_params:
            network_params.append({'params': base_params})
        if old_head_params:
            network_params.append({'params': old_head_params, 'lr_scale': old_head_lr_scale})

        if self._cur_task==0:
            if self.optim == 'sgd':
                for group in network_params:
                    group['lr'] = self.init_lr * group.pop('lr_scale', 1.0)
                optimizer = optim.SGD(params=network_params, momentum=0.9, lr=self.init_lr, weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.init_epoch)
            elif self.optim == 'adam':
                for group in network_params:
                    group['lr'] = self.init_lr * group.pop('lr_scale', 1.0)
                optimizer = optim.Adam(params=network_params, lr=self.init_lr, weight_decay=self.init_weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader,test_loader,optimizer,scheduler)
        else:
            if self.optim == 'sgd':
                for group in network_params:
                    group['lr'] = self.lrate * group.pop('lr_scale', 1.0)
                optimizer = optim.SGD(params=network_params, momentum=0.9, lr=self.lrate, weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=self.epochs)
            elif self.optim == 'adam':
                for group in network_params:
                    group['lr'] = self.lrate * group.pop('lr_scale', 1.0)
                optimizer = optim.Adam(params=network_params, lr=self.lrate, weight_decay=self.weight_decay, betas=(0.9,0.999))
                scheduler = CosineSchedule(optimizer=optimizer,K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler)


    def _task_label_offset(self, task_id: int) -> int:
        if task_id <= 0:
            return 0
        return int(sum(self.task_sizes[:task_id]))

    def _task_label_size(self, task_id: int) -> int:
        if task_id < 0 or task_id >= len(self.task_sizes):
            return 0
        return int(self.task_sizes[task_id])

    def _select_guidance_params(self):
        mode = str(self.args.get('distilled_guidance_target', 'head_lastlora')).lower()
        params = []
        last_blocks = max(1, int(self.args.get('distilled_guidance_last_blocks', 1)))
        block_start = max(0, 12 - last_blocks)
        for name, param in self._network.named_parameters():
            if not param.requires_grad:
                continue
            use = False
            if 'classifier_pool' in name:
                use = True
            elif mode in ('all', 'all_trainable'):
                use = True
            elif 'image_encoder.blocks.' in name and any(tok in name for tok in ('lora_', 'glora_', 'elora_')):
                match = re.search(r'image_encoder\.blocks\.(\d+)\.', name)
                if match is not None and int(match.group(1)) >= block_start:
                    use = True
            if use:
                params.append(param)
        if not params:
            params = [param for _, param in self._network.named_parameters() if param.requires_grad]
        return params

    @staticmethod
    def _flatten_grads_for_params(params, grads):
        flats = []
        for p, g in zip(params, grads):
            if g is None:
                flats.append(torch.zeros_like(p, memory_format=torch.contiguous_format).view(-1))
            else:
                flats.append(g.contiguous().view(-1))
        if not flats:
            return None
        return torch.cat(flats, dim=0)

    @staticmethod
    def _assign_flat_grad(params, flat_grad: torch.Tensor):
        offset = 0
        for p in params:
            numel = p.numel()
            grad = flat_grad[offset:offset + numel].view_as(p).clone()
            if p.grad is None:
                p.grad = grad
            else:
                p.grad.detach().copy_(grad)
            offset += numel

    @staticmethod
    def _project_simplex(v: torch.Tensor) -> torch.Tensor:
        if v.numel() == 1:
            return torch.ones_like(v)
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1.0
        ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
        cond = u - cssv / ind > 0
        rho = int(torch.nonzero(cond, as_tuple=False).max().item())
        theta = cssv[rho] / float(rho + 1)
        w = torch.clamp(v - theta, min=0.0)
        denom = w.sum().clamp_min(1e-12)
        return w / denom

    @staticmethod
    def _cosine_or_zero(a: torch.Tensor, b: torch.Tensor) -> float:
        if a is None or b is None or a.numel() == 0 or b.numel() == 0:
            return 0.0
        denom = a.norm() * b.norm()
        if float(denom.item()) <= 1e-12:
            return 0.0
        return float(torch.dot(a, b).item() / denom.item())

    @staticmethod
    def _min_norm_two_coeffs(v1: torch.Tensor, v2: torch.Tensor):
        v1v1 = float(torch.dot(v1, v1).item())
        v1v2 = float(torch.dot(v1, v2).item())
        v2v2 = float(torch.dot(v2, v2).item())
        if v1v2 >= v1v1:
            return 1.0, 0.0
        if v1v2 >= v2v2:
            return 0.0, 1.0
        denom = max(v1v1 + v2v2 - 2.0 * v1v2, 1e-12)
        gamma = (v2v2 - v1v2) / denom
        gamma = float(max(0.0, min(1.0, gamma)))
        return gamma, 1.0 - gamma

    def _pareto_simplex_min_norm(self, G: torch.Tensor) -> torch.Tensor:
        n = G.size(0)
        device, dtype = G.device, G.dtype
        if n == 1:
            return torch.ones(1, device=device, dtype=dtype)

        K = G @ G.t()
        # Pairwise min-norm initialization (closer to MGDA/simplex solvers than uniform init)
        best_val = None
        best_w = None
        for i in range(n):
            for j in range(i + 1, n):
                wi, wj = self._min_norm_two_coeffs(G[i], G[j])
                w = torch.zeros(n, device=device, dtype=dtype)
                w[i] = wi
                w[j] = wj
                val = float((w @ (K @ w)).item())
                if best_val is None or val < best_val:
                    best_val = val
                    best_w = w
        if best_w is None:
            best_w = torch.full((n,), 1.0 / float(n), device=device, dtype=dtype)

        weights = best_w
        lr = float(self.args.get('distilled_guidance_pareto_lr', 0.1))
        iters = int(self.args.get('distilled_guidance_pareto_steps', 50))
        tol = float(self.args.get('distilled_guidance_pareto_tol', 1e-6))
        use_adaptive = bool(self.args.get('distilled_guidance_pareto_adaptive_lr', True))

        # Optional normalized Gram matrix to avoid a single large-norm task dominating the simplex solution.
        if bool(self.args.get('distilled_guidance_pareto_normalize', False)):
            norms = torch.sqrt(torch.diag(K).clamp_min(1e-12))
            K_opt = K / (norms[:, None] * norms[None, :]).clamp_min(1e-12)
        else:
            K_opt = K

        prev_obj = float((weights @ (K_opt @ weights)).item())
        for _ in range(max(1, iters)):
            grad = 2.0 * (K_opt @ weights)
            if use_adaptive:
                denom = float(grad.norm().item())
                step = lr / max(denom, 1e-12)
            else:
                step = lr
            candidate = self._project_simplex(weights - step * grad)
            obj = float((candidate @ (K_opt @ candidate)).item())
            if prev_obj - obj < tol:
                weights = candidate
                break
            weights = candidate
            prev_obj = obj

        return weights

    def _aggregate_old_guidance(self, g_cur: torch.Tensor, old_grads, mode: str = 'uniform'):
        if len(old_grads) == 0:
            return None, None
        if len(old_grads) == 1:
            return old_grads[0][0], torch.ones(1, device=old_grads[0][0].device)
        G = torch.stack([g for g, _ in old_grads], dim=0)
        device = G.device
        mode = str(mode).lower()
        if mode == 'uniform':
            weights = torch.full((G.size(0),), 1.0 / float(G.size(0)), device=device, dtype=G.dtype)
        elif mode == 'conflict':
            tau = float(self.args.get('distilled_guidance_conflict_tau', 4.0))
            scores = []
            for g, _ in old_grads:
                scores.append(max(0.0, -self._cosine_or_zero(g_cur, g)))
            scores = torch.tensor(scores, device=device, dtype=G.dtype)
            if float(scores.sum().item()) <= 1e-12:
                weights = torch.full((G.size(0),), 1.0 / float(G.size(0)), device=device, dtype=G.dtype)
            else:
                weights = torch.softmax(tau * scores, dim=0)
        elif mode == 'pareto':
            weights = self._pareto_simplex_min_norm(G)
        else:
            weights = torch.full((G.size(0),), 1.0 / float(G.size(0)), device=device, dtype=G.dtype)
        agg = (weights.unsqueeze(1) * G).sum(dim=0)
        return agg, weights

    @staticmethod
    def _project_conflict(g_cur: torch.Tensor, g_old: torch.Tensor) -> torch.Tensor:
        if g_cur is None or g_old is None:
            return g_old
        dot = torch.dot(g_old, g_cur)
        if float(dot.item()) >= 0:
            return g_old
        denom = g_cur.pow(2).sum().clamp_min(1e-12)
        return g_old - dot / denom * g_cur

    def train_function(self, train_loader, test_loader, optimizer, scheduler):
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        enabled = set()
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        logging.info(f"Parameters to be updated: {enabled}")
        if self.replay_loader is not None:
            logging.info(f'[Distilled-Replay] Replay batches per epoch: {len(self.replay_loader)}')

        prog_bar = tqdm(range(self.run_epoch))

        loss_cos = AngularPenaltySMLoss(loss_type='cosface', s=self.scale, m=self.margin)
        if self._cur_task > 0 and self.args['cc'] is True:
            loss_maha = MahalanobisLoss(self._old_class_covs)

        replay_weight = float(self.args.get('distilled_replay_loss_weight', 1.0))
        replay_kd_weight = float(self.args.get('distilled_replay_kd_weight', 0.5))
        replay_temperature = float(self.args.get('distilled_replay_temperature', 2.0))
        replay_proto_weight = float(self.args.get('distilled_replay_proto_weight', 0.25))
        replay_new_suppress_weight = float(self.args.get('distilled_replay_new_suppress_weight', 0.2))
        replay_margin = float(self.args.get('distilled_replay_margin', 0.05))
        replay_patch_kd_weight = float(self.args.get('distilled_replay_patch_kd_weight', 0.0))
        replay_full_head = bool(self.args.get('distilled_replay_full_head', True))

        guidance_enable = bool(self.args.get('distilled_guidance_enable', False)) and self.replay_loader is not None and self._cur_task > 0
        guidance_beta = float(self.args.get('distilled_guidance_beta', 0.5))
        guidance_aux_weight = float(self.args.get('distilled_guidance_aux_weight', 0.0))
        guidance_mode = str(self.args.get('distilled_guidance_mode', 'uniform')).lower()
        guidance_kd_weight = float(self.args.get('distilled_guidance_kd_weight', 0.0))
        guidance_project_conflict = bool(self.args.get('distilled_guidance_project_conflict', False))
        guidance_params = self._select_guidance_params() if guidance_enable else []
        if guidance_enable:
            logging.info(f'[Distilled-Guidance] mode={guidance_mode}, beta={guidance_beta:.3f}, aux={guidance_aux_weight:.3f}, kd={guidance_kd_weight:.3f}, project={guidance_project_conflict}, target_params={len(guidance_params)}')

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            replay_iter = iter(self.replay_loader) if self.replay_loader is not None else None
            replay_loss_meter = 0.0
            replay_kd_meter = 0.0
            replay_proto_meter = 0.0
            replay_suppress_meter = 0.0
            replay_patch_meter = 0.0
            replay_steps = 0
            guidance_cos_meter = 0.0
            guidance_old_norm_meter = 0.0
            guidance_weight_max_meter = 0.0
            guidance_weight_entropy_meter = 0.0
            guidance_steps = 0
            cls_meter = 0.0
            cc_meter = 0.0
            patch_sd_meter = 0.0
            main_steps = 0

            for _, batch in enumerate(train_loader):
                _, inputs, targets = batch
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                local_targets = targets - self._known_classes

                output = self._network(inputs)
                logits = output['logits']
                features = output['features']
                patch_tokens = output['patch_tokens']
                cls_loss = loss_cos(logits, local_targets)
                loss = cls_loss
                cls_meter += float(cls_loss.detach().item())
                main_steps += 1

                if self._cur_task > 0 and self.args['cc'] is True:
                    with torch.no_grad():
                        old_output = self._old_network(inputs)
                        old_features = old_output['features']
                        old_patch_tokens = old_output['patch_tokens']
                    cc_loss = loss_maha(old_features, features, local_targets)
                    patch_sd_loss = compute_angle_weighted_patch_distillation_loss(patch_tokens, old_patch_tokens, features)
                    cc_meter += float(cc_loss.detach().item())
                    patch_sd_meter += float(patch_sd_loss.detach().item())
                    loss = loss + cc_loss
                    loss = loss + self.args['lamb_p'] * patch_sd_loss

                g_cur = None
                if guidance_enable and guidance_params:
                    g_cur_list = torch.autograd.grad(loss, guidance_params, retain_graph=True, allow_unused=True)
                    g_cur = self._flatten_grads_for_params(guidance_params, g_cur_list)

                replay_aux_objective = None
                g_old = None
                if replay_iter is not None:
                    try:
                        replay_batch = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(self.replay_loader)
                        replay_batch = next(replay_iter)

                    if len(replay_batch) == 4:
                        _, replay_inputs, replay_targets, replay_task_ids = replay_batch
                    else:
                        _, replay_inputs, replay_targets = replay_batch
                        replay_task_ids = torch.full_like(replay_targets, -1)

                    replay_inputs = replay_inputs.to(self._device)
                    replay_targets = replay_targets.to(self._device)
                    replay_task_ids = replay_task_ids.to(self._device)
                    replay_steps += 1

                    if guidance_enable and guidance_params:
                        replay_global_logits = self._network.interface(replay_inputs)
                        grouped_losses = []
                        for task_tensor in torch.unique(replay_task_ids):
                            task_id = int(task_tensor.item())
                            mask = replay_task_ids == task_tensor
                            if int(mask.sum().item()) <= 0:
                                continue
                            offset = self._task_label_offset(task_id) if task_id >= 0 else 0
                            task_size = self._task_label_size(task_id) if task_id >= 0 else self._known_classes
                            if task_size <= 0:
                                continue
                            task_logits = replay_global_logits[mask, offset:offset + task_size]
                            task_targets = replay_targets[mask] - offset
                            task_loss = loss_cos(task_logits, task_targets)
                            replay_loss_meter += float(task_loss.item())
                            if self._old_network is not None and guidance_kd_weight > 0:
                                with torch.no_grad():
                                    teacher_logits = self._old_network.interface(replay_inputs[mask])[:, offset:offset + task_size]
                                kd_loss = self._logit_kd_loss(task_logits, teacher_logits, temperature=replay_temperature)
                                task_loss = task_loss + guidance_kd_weight * kd_loss
                                replay_kd_meter += float(kd_loss.item())
                            grouped_losses.append((task_id, task_loss))

                        old_grads = []
                        for task_id, task_loss in grouped_losses:
                            g_i_list = torch.autograd.grad(task_loss, guidance_params, retain_graph=True, allow_unused=True)
                            g_i = self._flatten_grads_for_params(guidance_params, g_i_list)
                            if g_i is not None:
                                old_grads.append((g_i, task_id))
                        if g_cur is not None and old_grads:
                            g_old, g_weights = self._aggregate_old_guidance(g_cur, old_grads, mode=guidance_mode)
                            if guidance_project_conflict:
                                g_old = self._project_conflict(g_cur, g_old)
                            guidance_cos_meter += self._cosine_or_zero(g_cur, g_old)
                            guidance_old_norm_meter += float(g_old.norm().item())
                            if g_weights is not None:
                                guidance_weight_max_meter += float(g_weights.max().item())
                                ent = -(g_weights * torch.log(g_weights.clamp_min(1e-12))).sum()
                                guidance_weight_entropy_meter += float(ent.item())
                            guidance_steps += 1

                    if (not guidance_enable) or guidance_aux_weight > 0:
                        replay_global_logits = self._network.interface(replay_inputs)
                        replay_student_old_logits = replay_global_logits[:, :self._known_classes]
                        ce_logits = replay_global_logits if replay_full_head else replay_student_old_logits
                        loss_replay = loss_cos(ce_logits, replay_targets)
                        replay_objective = replay_weight * loss_replay
                        if not guidance_enable:
                            replay_loss_meter += float(loss_replay.item())

                        replay_features = replay_patch_tokens = None
                        need_replay_tokens = (replay_proto_weight > 0 and self._class_means is not None) or replay_patch_kd_weight > 0
                        if need_replay_tokens:
                            replay_features, replay_patch_tokens = self._network.extract_tokens(replay_inputs)

                        if self._old_network is not None and replay_kd_weight > 0:
                            with torch.no_grad():
                                teacher_logits = self._old_network.interface(replay_inputs)
                            kd_loss = self._logit_kd_loss(replay_student_old_logits, teacher_logits, temperature=replay_temperature)
                            replay_objective = replay_objective + replay_kd_weight * kd_loss
                            if not guidance_enable:
                                replay_kd_meter += float(kd_loss.item())

                        if replay_proto_weight > 0 and self._class_means is not None and replay_features is not None:
                            class_means = self._class_means
                            if not torch.is_tensor(class_means):
                                class_means = torch.tensor(class_means, dtype=replay_features.dtype)
                            proto_targets = class_means[replay_targets.detach().cpu()].to(self._device, dtype=replay_features.dtype)
                            proto_loss = F.mse_loss(F.normalize(replay_features, p=2, dim=1), F.normalize(proto_targets, p=2, dim=1))
                            replay_objective = replay_objective + replay_proto_weight * proto_loss
                            if not guidance_enable:
                                replay_proto_meter += float(proto_loss.item())

                        if replay_new_suppress_weight > 0 and replay_global_logits.size(1) > self._known_classes:
                            target_scores = replay_global_logits.gather(1, replay_targets.unsqueeze(1)).squeeze(1)
                            max_new_scores = replay_global_logits[:, self._known_classes:].max(dim=1).values
                            suppress_loss = F.relu(max_new_scores - target_scores + replay_margin).mean()
                            replay_objective = replay_objective + replay_new_suppress_weight * suppress_loss
                            if not guidance_enable:
                                replay_suppress_meter += float(suppress_loss.item())

                        if self._old_network is not None and replay_patch_kd_weight > 0:
                            if replay_features is None or replay_patch_tokens is None:
                                replay_features, replay_patch_tokens = self._network.extract_tokens(replay_inputs)
                            with torch.no_grad():
                                old_replay_features, old_replay_patch = self._old_network.extract_tokens(replay_inputs)
                            patch_kd_loss = compute_angle_weighted_patch_distillation_loss(replay_patch_tokens, old_replay_patch, replay_features)
                            replay_objective = replay_objective + replay_patch_kd_weight * patch_kd_loss
                            if not guidance_enable:
                                replay_patch_meter += float(patch_kd_loss.item())

                        replay_aux_objective = replay_objective

                optimizer.zero_grad()
                loss.backward()
                if guidance_enable:
                    if replay_aux_objective is not None and guidance_aux_weight > 0:
                        (guidance_aux_weight * replay_aux_objective).backward()
                    if guidance_params and g_cur is not None and g_old is not None:
                        final_guidance = g_cur + guidance_beta * g_old
                        self._assign_flat_grad(guidance_params, final_guidance)
                        step_loss_value = float(loss.item())
                    else:
                        step_loss_value = float(loss.item())
                else:
                    step_loss_value = float(loss.item())
                    if replay_aux_objective is not None:
                        replay_aux_objective.backward()
                        step_loss_value += float(replay_aux_objective.item())

                optimizer.step()
                losses += step_loss_value

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(local_targets.expand_as(preds)).cpu().sum()
                total += len(local_targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc)
            if main_steps > 0:
                info += ', Cls {:.3f}, CC {:.3f}, PatchSD {:.3f}, PatchSD_w {:.3f}'.format(
                    cls_meter / main_steps,
                    cc_meter / main_steps,
                    patch_sd_meter / main_steps,
                    float(self.args.get('lamb_p', 0.0)) * patch_sd_meter / main_steps,
                )
            if replay_steps > 0:
                info += ', Replay_loss {:.3f}, Replay_kd {:.3f}, Replay_proto {:.3f}, Replay_sup {:.3f}, Replay_patch {:.3f}'.format(
                    replay_loss_meter / replay_steps,
                    replay_kd_meter / replay_steps if replay_steps > 0 else 0.0,
                    replay_proto_meter / replay_steps if replay_steps > 0 else 0.0,
                    replay_suppress_meter / replay_steps if replay_steps > 0 else 0.0,
                    replay_patch_meter / replay_steps if replay_steps > 0 else 0.0,
                )
            if guidance_steps > 0:
                info += ', Guide_cos {:.3f}, Guide_norm {:.3f}, Guide_wmax {:.3f}, Guide_went {:.3f}'.format(guidance_cos_meter / guidance_steps, guidance_old_norm_meter / guidance_steps, guidance_weight_max_meter / guidance_steps, guidance_weight_entropy_meter / guidance_steps)
            prog_bar.set_description(info)

        test_acc = self._compute_accuracy(self._network, test_loader)
        final_info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(self._cur_task, epoch + 1, self.run_epoch, losses / len(train_loader), train_acc, test_acc)
        if main_steps > 0:
            final_info += ', Cls {:.3f}, CC {:.3f}, PatchSD {:.3f}, PatchSD_w {:.3f}'.format(
                cls_meter / main_steps,
                cc_meter / main_steps,
                patch_sd_meter / main_steps,
                float(self.args.get('lamb_p', 0.0)) * patch_sd_meter / main_steps,
            )
        if replay_steps > 0:
            final_info += ', Replay_loss {:.3f}, Replay_kd {:.3f}, Replay_proto {:.3f}, Replay_sup {:.3f}, Replay_patch {:.3f}'.format(
                replay_loss_meter / replay_steps,
                replay_kd_meter / replay_steps if replay_steps > 0 else 0.0,
                replay_proto_meter / replay_steps if replay_steps > 0 else 0.0,
                replay_suppress_meter / replay_steps if replay_steps > 0 else 0.0,
                replay_patch_meter / replay_steps if replay_steps > 0 else 0.0,
            )
        if guidance_steps > 0:
            final_info += ', Guide_cos {:.3f}, Guide_norm {:.3f}, Guide_wmax {:.3f}, Guide_went {:.3f}'.format(guidance_cos_meter / guidance_steps, guidance_old_norm_meter / guidance_steps, guidance_weight_max_meter / guidance_steps, guidance_weight_entropy_meter / guidance_steps)
        logging.info(final_info)

    def accuracy(self, y_pred, y_true, accuracy_matrix = False):
        assert len(y_pred) == len(y_true), 'Data length error.'
        all_acc = {}
        all_acc['total'] = np.around((y_pred == y_true).sum()*100 / len(y_true), decimals=2)
        
        i = 0
        # Grouped accuracy
        for class_id in range(0, np.max(y_true), self.class_num):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + self.class_num))[0]
            label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id+self.class_num-1).rjust(2, '0'))
            all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)
            if accuracy_matrix:
                self.acc_matrix[i, self._cur_task] = all_acc[label] 
            i += 1

        # Old accuracy
        idxes = np.where(y_true < self._known_classes)[0]
        all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes),
                                                            decimals=2)

        # New accuracy
        idxes = np.where(y_true >= self._known_classes)[0]
        all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum()*100 / len(idxes), decimals=2)

        return all_acc

    def _evaluate(self, y_pred, y_true, accuracy_matrix=False):
        ret = {}
        # print(len(y_pred), len(y_true))
        grouped = self.accuracy(y_pred, y_true, accuracy_matrix=accuracy_matrix)
        ret['grouped'] = grouped
        ret['top1'] = grouped['total']
        return ret

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_with_task = []
        y_pred_task, y_true_task = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                task_id = (targets//self.class_num).cpu()
                y_true_task.append(task_id)

                outputs = self._network.interface(inputs)

            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
            y_pred_task.append((predicts//self.class_num).cpu())

            outputs_with_task = torch.zeros_like(outputs)[:,:self.class_num]
            for idx, i in enumerate(targets//self.class_num):
                en, be = self.class_num*i, self.class_num*(i+1)
                outputs_with_task[idx] = outputs[idx, en:be]
            predicts_with_task = outputs_with_task.argmax(dim=1)
            predicts_with_task = predicts_with_task + (targets//self.class_num)*self.class_num

            y_pred.append(predicts.cpu().numpy())
            y_pred_with_task.append(predicts_with_task.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(y_pred_task), torch.cat(y_true_task)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.interface(inputs)
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        for p in self._network.classifier_pool[:self._cur_task+1].parameters():
            p.requires_grad=True
            
        run_epochs = ca_epochs
        crct_num = self._total_classes    
        param_list = [p for p in self._network.classifier_pool.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': 0.01,
                           'weight_decay': 0.0005}]
        optimizer = optim.SGD(network_params, lr=0.01, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
        
            for c_id in range(crct_num):
                t_id = c_id//task_size
                decay = (t_id+1)/(self._cur_task+1)*0.1
                cls_mean = self._class_means[c_id].to(self._device)*(0.9+decay)
                cls_cov = self._class_covs[c_id].to(self._device)

                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)                
                sampled_label.extend([c_id]*num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]
            
            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                # -stage two only use classifiers
                outputs = self._network(inp, fc_only=True)
                logits = outputs

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses/self._total_classes, test_acc)
            logging.info(info)


    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = torch.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = torch.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        for class_idx in range(self._known_classes, self._total_classes):

            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            class_mean = torch.mean(torch.tensor(vectors), dim=0)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean.detach()
            self._class_covs[class_idx, ...] = class_cov.detach()

    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2 - Y1
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement
    
    def extract_features(self, trainloader, model, task_id = None):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data, task_id)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list
    
    def _extract_vectors_adv(self, loader, old=False):
        if old:
            network = self._old_network
        else:
            network = self._network
        network.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                (_,_inputs, _targets) = batch
                _inputs = _inputs.to(self._device)
                _vectors = network.extract_vector(_inputs)
                vectors.append(_vectors)
                targets.append(_targets)

        return torch.cat(vectors, dim=0), torch.cat(targets, dim=0)


    def shrink_cov(self, cov):
        alpha1 = 10
        alpha2 = 10
        # Compute the mean of the diagonal elements
        diag_mean = torch.mean(torch.diagonal(cov))
        
        # Create a copy of the covariance matrix with zeroed out diagonals
        off_diag = cov.clone()
        off_diag.fill_diagonal_(0.0)
        
        # Compute the mean of the off-diagonal elements (non-zero entries)
        mask = off_diag != 0.0
        if int(mask.sum().item()) > 0:
            off_diag_mean = (off_diag * mask).sum() / mask.sum()
        else:
            off_diag_mean = cov.new_zeros(())
        
        # Identity matrix
        iden = torch.eye(cov.size(0), device=cov.device)
        
        # Shrink the covariance matrix
        cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))
        
        return cov_
    
    def _compute_class_invcov(self, data_manager):
        _class_invcovs = torch.zeros((self.class_num, self.feature_dim, self.feature_dim),device=self._device)

        for class_idx in range(self._known_classes, self._total_classes):

            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors_adv(idx_loader, True)

            class_cov = self.shrink_cov(torch.cov(torch.tensor(vectors, dtype=torch.float64).T)) + torch.eye(self.feature_dim).to(self._device) * 1e-3
            _class_invcovs[class_idx-self._known_classes, ...] = torch.linalg.pinv(class_cov).detach()

        return _class_invcovs
