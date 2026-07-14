import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image


def tensor_to_uint8_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
    return x


def save_image_tensor(x: torch.Tensor, path: Union[str, Path]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8_image(x)).save(path)


def save_grid(images: torch.Tensor, path: Union[str, Path], pad: int = 2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    if n == 0:
        return
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w = images.shape[-2:]
    canvas = np.ones((rows * h + pad * (rows - 1), cols * w + pad * (cols - 1), 3), dtype=np.uint8) * 255
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0:y0+h, x0:x0+w] = tensor_to_uint8_image(img)
    Image.fromarray(canvas).save(path)


def save_manifest(path: Union[str, Path], **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(kwargs, f, indent=2, ensure_ascii=False)


def save_distilled_bundle(images: torch.Tensor, labels: torch.Tensor, out_dir: Union[str, Path], step: Optional[int] = None, **metadata):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'images': images.detach().cpu(), 'labels': labels.detach().cpu()}, out_dir/'data.pth')
    save_grid(images, out_dir/'grid.png')
    per_class = out_dir/'per_class'
    per_class.mkdir(exist_ok=True)
    counts = {}
    for img, label in zip(images, labels):
        cls = int(label.item())
        counts.setdefault(cls, 0)
        idx = counts[cls]
        counts[cls] += 1
        save_image_tensor(img, per_class/f'class_{cls:03d}_ipc{idx}.png')
    manifest = dict(num_images=int(len(images)), labels=[int(x) for x in labels.detach().cpu().tolist()], step=step)
    manifest.update(metadata)
    save_manifest(out_dir/'manifest.json', **manifest)


@contextmanager
def temporarily_freeze_model(model):
    if model is None:
        yield
        return

    params = list(model.parameters())
    requires_grad_states = [p.requires_grad for p in params]
    was_training = model.training
    try:
        model.zero_grad(set_to_none=True)
    except TypeError:
        model.zero_grad()
    for param in params:
        param.requires_grad_(False)
    model.eval()
    try:
        yield model
    finally:
        try:
            model.zero_grad(set_to_none=True)
        except TypeError:
            model.zero_grad()
        for param, state in zip(params, requires_grad_states):
            param.requires_grad_(state)
        if was_training:
            model.train()
        else:
            model.eval()
        try:
            model.zero_grad(set_to_none=True)
        except TypeError:
            model.zero_grad()


@contextmanager
def temporarily_set_grad_checkpointing(model, enable=True):
    if model is None:
        yield
        return

    target = None
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'set_grad_checkpointing'):
        target = model.image_encoder
    elif hasattr(model, 'set_grad_checkpointing'):
        target = model

    if target is None:
        yield
        return

    prev = bool(getattr(target, 'grad_checkpointing', False))
    target.set_grad_checkpointing(enable)
    try:
        yield model
    finally:
        target.set_grad_checkpointing(prev)
