from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BaseDistilledDataset
from .utils import save_grid


class PyramidDataset(BaseDistilledDataset):
    """Port closely following official src/synsets/pyramid.py.

    This variant additionally supports per-class variable IPC so hard classes can
    be expanded later without increasing synthetic images for every class.
    """

    def __init__(self, num_classes: int, cfg, device: torch.device, out_dir: str, init_synset_now: bool = True):
        super().__init__(device=device)
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.out_dir = out_dir
        self.pyramid = []
        self.syn_labels = torch.empty(0, dtype=torch.long, device=self.device)
        self.class_ipc = self._resolve_class_ipc()
        if init_synset_now:
            self.pyramid, self.syn_labels = self.init_synset()
        self.optimizer = self.init_optimizer() if self.pyramid else None

    def _resolve_class_ipc(self) -> List[int]:
        class_ipc = getattr(self.cfg, 'class_ipc', None)
        if class_ipc is None:
            return [int(self.cfg.ipc) for _ in range(self.num_classes)]
        class_ipc = [max(1, int(x)) for x in class_ipc]
        if len(class_ipc) < self.num_classes:
            pad = class_ipc[-1] if class_ipc else int(self.cfg.ipc)
            class_ipc = class_ipc + [int(pad)] * (self.num_classes - len(class_ipc))
        return class_ipc[: self.num_classes]

    def _make_syn_labels(self, class_ipc: Optional[List[int]] = None) -> Tensor:
        ipc_list = self.class_ipc if class_ipc is None else [max(1, int(x)) for x in class_ipc]
        labels = []
        for c, ipc in enumerate(ipc_list):
            labels.extend([c] * int(ipc))
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def get_class_ipc(self) -> List[int]:
        return [int(x) for x in self.class_ipc]

    def get_class_counts_tensor(self) -> Tensor:
        counts = torch.bincount(self.syn_labels, minlength=self.num_classes)
        if counts.numel() < self.num_classes:
            pad = torch.zeros(self.num_classes - counts.numel(), dtype=counts.dtype, device=counts.device)
            counts = torch.cat([counts, pad], dim=0)
        return counts[: self.num_classes]

    def init_optimizer(self):
        if not self.pyramid:
            return None
        optimizer = torch.optim.Adam([{'params': p, 'lr': self.cfg.lr} for p in self.pyramid])
        return optimizer

    def init_synset(self) -> Tuple[List[Tensor], Tensor]:
        self.class_ipc = self._resolve_class_ipc()
        syn_labels = self._make_syn_labels(self.class_ipc)
        num_images = len(syn_labels)
        pyramid = []
        res = 1
        while res <= self.cfg.pyramid_start_res:
            level = torch.randn((num_images, 3, res, res), device=self.device)
            if self.cfg.init_mode == 'zero':
                level = level * 0
            pyramid.insert(0, level)
            res *= 2
            if res > self.cfg.syn_res:
                res = self.cfg.syn_res
        pyramid = [p / len(pyramid) for p in pyramid]
        for p in pyramid:
            p.requires_grad_(True)
        return pyramid, syn_labels

    def extend_pyramid(self) -> bool:
        old_len = len(self.pyramid)
        new_len = old_len + 1
        old_res = self.pyramid[0].shape[-1]
        if old_res == self.cfg.syn_res:
            return False
        new_res = min(old_res * 2, self.cfg.syn_res)
        num_images = self.pyramid[-1].shape[0]
        self.pyramid = [p.detach().clone() * old_len / new_len for p in self.pyramid]
        if self.cfg.init_mode == 'zero':
            new_layer = torch.sum(torch.stack([F.interpolate(p, (new_res, new_res), antialias=False, mode='bilinear') for p in self.pyramid]), dim=0)
            new_layer = new_layer / old_len
        else:
            new_layer = torch.randn((num_images, 3, new_res, new_res), device=self.device) / new_len
        self.pyramid.insert(0, new_layer)
        for p in self.pyramid:
            p.requires_grad_(True)
        self.optimizer = self.init_optimizer()
        return True

    def expand_classes(self, class_ids: List[int], extra_ipc: int = 1, max_ipc: Optional[int] = None) -> List[int]:
        if not self.pyramid or extra_ipc <= 0:
            return []
        target_classes = []
        class_ipc = self.get_class_ipc()
        additions_by_class = {}
        for cls in sorted({int(c) for c in class_ids if 0 <= int(c) < self.num_classes}):
            cur_ipc = int(class_ipc[cls])
            tgt_ipc = cur_ipc + int(extra_ipc)
            if max_ipc is not None:
                tgt_ipc = min(tgt_ipc, int(max_ipc))
            add_n = max(0, tgt_ipc - cur_ipc)
            if add_n <= 0:
                continue
            source_idx = torch.nonzero(self.syn_labels == cls, as_tuple=False).flatten()
            if source_idx.numel() == 0:
                continue
            pick = source_idx[torch.arange(add_n, device=self.device) % source_idx.numel()]
            additions_by_class[cls] = pick
            target_classes.append(cls)
            class_ipc[cls] = tgt_ipc
        if not target_classes:
            return []

        new_pyramid = []
        level_scale = max(1, len(self.pyramid))
        for level in self.pyramid:
            chunks = [level.detach().clone()]
            for cls in target_classes:
                source_idx = additions_by_class[cls]
                extra = level.index_select(0, source_idx).detach().clone()
                if self.cfg.init_mode != 'zero':
                    noise = torch.randn_like(extra) * (0.01 / float(level_scale))
                    extra = extra + noise
                chunks.append(extra)
            merged = torch.cat(chunks, dim=0)
            merged.requires_grad_(True)
            new_pyramid.append(merged)
        self.pyramid = new_pyramid

        new_labels = [self.syn_labels.detach().clone()]
        for cls in target_classes:
            add_n = int(additions_by_class[cls].numel())
            new_labels.append(torch.full((add_n,), int(cls), dtype=torch.long, device=self.device))
        self.syn_labels = torch.cat(new_labels, dim=0)
        self.class_ipc = class_ipc
        self.optimizer = self.init_optimizer()
        return target_classes

    def decode_pyramid(self, indices: Optional[Tensor] = None) -> Tensor:
        levels = self.pyramid if indices is None else [p[indices] for p in self.pyramid]
        result = torch.sum(torch.stack([F.interpolate(p, (self.cfg.syn_res, self.cfg.syn_res), antialias=False, mode='bilinear') for p in levels]), dim=0)
        if self.cfg.decorrelate_color:
            result = self.linear_decorrelate_color(result)
        result = torch.sigmoid(2 * result)
        return result

    def get_data(self) -> Tuple[Tensor, Tensor]:
        syn_images = self.decode_pyramid()
        return syn_images, self.syn_labels

    def get_data_by_indices(self, indices: Tensor) -> Tuple[Tensor, Tensor]:
        idx = indices.to(self.device)
        syn_images = self.decode_pyramid(indices=idx)
        return syn_images, self.syn_labels[idx]

    @torch.no_grad()
    def log_images(self, step: Optional[int] = None):
        if len(self.pyramid[0]) > 256:
            return
        syn_images, _ = self.get_data()
        if step is None:
            step_name = 'latest'
        else:
            step_name = f'{step:05d}'
        save_grid(syn_images.detach().cpu(), f'{self.out_dir}/images_{step_name}.png')

    def upkeep(self, step: Optional[int] = None):
        if step is not None and step > 1 and (step % self.cfg.pyramid_extent_it == 0):
            self.extend_pyramid()

    def load_from_images(self, images: Tensor, labels: Optional[Tensor] = None):
        if labels is not None:
            self.syn_labels = labels.detach().to(self.device).long()
            counts = torch.bincount(self.syn_labels, minlength=self.num_classes)
            self.class_ipc = [int(x) for x in counts[: self.num_classes].tolist()]
        images = images.detach().to(self.device).float().clamp(1e-4, 1 - 1e-4)
        if len(self.pyramid) != 1 or self.pyramid[0].shape[-1] != self.cfg.syn_res:
            self.pyramid = [torch.zeros_like(images, device=self.device)]
        raw = torch.log(images / (1 - images)) / 2.0
        with torch.no_grad():
            self.pyramid[0].copy_(raw)
        self.pyramid[0].requires_grad_(True)
        self.optimizer = self.init_optimizer()

    def get_save_dict(self):
        opt_state = self.optimizer.state_dict() if self.optimizer is not None else None
        return {
            'pyramid': [p.detach().cpu() for p in self.pyramid],
            'opt_state': opt_state,
            'syn_labels': self.syn_labels.detach().cpu(),
            'class_ipc': self.get_class_ipc(),
        }

    def load_from_dict(self, load_dict: dict):
        loaded_pyramid = load_dict['pyramid']
        need_rebuild = False
        if not self.pyramid or len(self.pyramid) != len(loaded_pyramid):
            need_rebuild = True
        else:
            for p, loaded_p in zip(self.pyramid, loaded_pyramid):
                if tuple(p.shape) != tuple(loaded_p.shape):
                    need_rebuild = True
                    break
        if need_rebuild:
            self.pyramid = []
            for loaded_p in loaded_pyramid:
                param = loaded_p.detach().to(self.device).clone()
                param.requires_grad_(True)
                self.pyramid.append(param)
        else:
            with torch.no_grad():
                for p, loaded_p in zip(self.pyramid, loaded_pyramid):
                    p.copy_(loaded_p.to(self.device))
        saved_labels = load_dict.get('syn_labels', None)
        saved_class_ipc = load_dict.get('class_ipc', None)
        if saved_labels is not None:
            self.syn_labels = saved_labels.detach().to(self.device).long()
            counts = torch.bincount(self.syn_labels, minlength=self.num_classes)
            self.class_ipc = [int(x) for x in counts[: self.num_classes].tolist()]
        elif saved_class_ipc is not None:
            self.class_ipc = [max(1, int(x)) for x in saved_class_ipc][: self.num_classes]
            self.syn_labels = self._make_syn_labels(self.class_ipc)
        else:
            self.class_ipc = self._resolve_class_ipc()
            self.syn_labels = self._make_syn_labels(self.class_ipc)
        self.optimizer = self.init_optimizer()
        opt_state = load_dict.get('opt_state', None)
        if opt_state is not None and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception:
                pass
