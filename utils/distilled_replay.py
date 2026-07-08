from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class DistilledReplayDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, trsf: Optional[Callable] = None, task_ids: Optional[torch.Tensor] = None):
        assert len(images) == len(labels), 'Data size error!'
        self.images = images.detach().cpu().float()
        self.labels = labels.detach().cpu().long()
        if task_ids is None:
            task_ids = torch.full((len(self.labels),), -1, dtype=torch.long)
        self.task_ids = torch.as_tensor(task_ids).detach().cpu().long().view(-1)
        if len(self.task_ids) != len(self.labels):
            raise ValueError('Task-id count does not match number of replay labels')
        self.trsf = trsf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].clamp(0, 1)
        image = Image.fromarray((image.permute(1, 2, 0).numpy() * 255.0).round().astype('uint8'))
        if self.trsf is not None:
            image = self.trsf(image)
        label = int(self.labels[idx].item())
        task_id = int(self.task_ids[idx].item())
        return idx, image, label, task_id


def _find_bundle(task_dir: Path) -> Optional[Path]:
    # Prefer canonical exports over transient checkpoints.
    candidates = [
        task_dir / 'data.pth',
        task_dir / 'final' / 'data.pth',
        task_dir / 'latest' / 'data.pth',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _canonicalize_images(images: torch.Tensor) -> torch.Tensor:
    images = torch.as_tensor(images).detach().cpu().float()

    # Accept a variety of legacy shapes and normalize to [N, 3, H, W].
    if images.ndim == 2:  # [H, W]
        images = images.unsqueeze(0).unsqueeze(0)
    elif images.ndim == 3:
        if images.shape[0] in (1, 3):  # [C, H, W]
            images = images.unsqueeze(0)
        elif images.shape[-1] in (1, 3):  # [H, W, C]
            images = images.permute(2, 0, 1).unsqueeze(0)
        else:  # [N, H, W]
            images = images.unsqueeze(1)
    elif images.ndim == 4:
        if images.shape[1] in (1, 3):  # [N, C, H, W]
            pass
        elif images.shape[-1] in (1, 3):  # [N, H, W, C]
            images = images.permute(0, 3, 1, 2)
        else:
            raise ValueError(f'Unsupported 4D distilled replay tensor shape: {tuple(images.shape)}')
    else:
        raise ValueError(f'Unsupported distilled replay tensor rank: {images.ndim}')

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.shape[1] != 3:
        raise ValueError(f'Expected 1 or 3 channels, got shape {tuple(images.shape)}')

    return images.contiguous()


def _canonicalize_labels(labels: torch.Tensor, num_images: int) -> torch.Tensor:
    labels = torch.as_tensor(labels).detach().cpu().long().view(-1)
    if labels.numel() == 1 and num_images > 1:
        labels = labels.repeat(num_images)
    if labels.numel() != num_images:
        raise ValueError(f'Label count {labels.numel()} does not match number of images {num_images}')
    return labels


def load_distilled_replay(
    base_dir: Path,
    upto_task_exclusive: int,
    task_sizes: Sequence[int],
    return_task_ids: bool = False,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], List[Path]]:
    images_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    task_id_list: List[torch.Tensor] = []
    used_paths: List[Path] = []

    class_offset = 0
    target_hw: Optional[Tuple[int, int]] = None
    for task_id in range(upto_task_exclusive):
        task_dir = base_dir / f'task_{task_id:02d}'
        bundle_path = _find_bundle(task_dir)
        if bundle_path is None:
            class_offset += int(task_sizes[task_id])
            continue

        payload = torch.load(bundle_path, map_location='cpu')
        images = _canonicalize_images(payload['images'])
        labels = _canonicalize_labels(payload['labels'], images.shape[0]) + int(class_offset)

        if target_hw is None:
            target_hw = tuple(images.shape[-2:])
        elif tuple(images.shape[-2:]) != target_hw:
            images = F.interpolate(images, size=target_hw, mode='bilinear', align_corners=False)

        images_list.append(images)
        labels_list.append(labels)
        task_id_list.append(torch.full((images.shape[0],), int(task_id), dtype=torch.long))
        used_paths.append(bundle_path)
        class_offset += int(task_sizes[task_id])

    if not images_list:
        if return_task_ids:
            return None, None, None, used_paths
        return None, None, used_paths

    images = torch.cat(images_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    task_ids = torch.cat(task_id_list, dim=0) if task_id_list else None
    if return_task_ids:
        return images, labels, task_ids, used_paths
    return images, labels, used_paths
