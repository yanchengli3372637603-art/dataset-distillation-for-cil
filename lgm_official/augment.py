import math
import random

import torch
import torch.nn.functional as F


class IdentityAugmentor:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def paired(self, x_a: torch.Tensor, x_b: torch.Tensor):
        return x_a, x_b


class StandardAugmentor:
    """Differentiable approximation of the training-time image pipeline.

    This mirrors the project's train transform more closely than the earlier
    square-crop approximation by sampling RandomResizedCrop-style scale and
    aspect ratio, then applying horizontal flip, all with PyTorch ops so
    gradients can still flow to synthetic images.

    In addition, ``paired`` applies the same geometric augmentation parameters
    to two aligned batches. This is useful for drift-aware refresh, where the
    teacher and student synthetic images should be compared under the same view.
    """

    def __init__(
        self,
        crop_res: int,
        scale=(0.05, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip_p: float = 0.5,
        noise_std: float = 0.0,
    ):
        self.crop_res = crop_res
        self.scale = scale
        self.ratio = ratio
        self.hflip_p = hflip_p
        self.noise_std = noise_std

    def _sample_crop(self, h: int, w: int):
        area = float(h * w)
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            aspect = math.exp(random.uniform(log_ratio[0], log_ratio[1]))
            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))
            if 0 < crop_w <= w and 0 < crop_h <= h:
                y0 = 0 if h == crop_h else random.randint(0, h - crop_h)
                x0 = 0 if w == crop_w else random.randint(0, w - crop_w)
                return y0, x0, crop_h, crop_w

        in_ratio = float(w) / float(h)
        if in_ratio < self.ratio[0]:
            crop_w = w
            crop_h = int(round(crop_w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            crop_h = h
            crop_w = int(round(crop_h * self.ratio[1]))
        else:
            crop_h = h
            crop_w = w
        y0 = max((h - crop_h) // 2, 0)
        x0 = max((w - crop_w) // 2, 0)
        return y0, x0, crop_h, crop_w

    def _sample_params(self, h: int, w: int):
        y0, x0, crop_h, crop_w = self._sample_crop(h, w)
        do_flip = random.random() < self.hflip_p
        return {
            'y0': y0,
            'x0': x0,
            'crop_h': crop_h,
            'crop_w': crop_w,
            'do_flip': do_flip,
        }

    def _apply_one(self, img: torch.Tensor, params, noise: torch.Tensor = None) -> torch.Tensor:
        y0 = int(params['y0'])
        x0 = int(params['x0'])
        crop_h = int(params['crop_h'])
        crop_w = int(params['crop_w'])
        img = img[:, y0:y0 + crop_h, x0:x0 + crop_w]
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.crop_res, self.crop_res),
            mode='bilinear',
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        if bool(params['do_flip']):
            img = torch.flip(img, dims=[2])
        if self.noise_std > 0:
            if noise is None:
                noise = torch.randn_like(img)
            img = (img + noise * self.noise_std).clamp(0, 1)
        return img

    def _crop_one(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        params = self._sample_params(h, w)
        noise = torch.randn_like(img[:, :self.crop_res, :self.crop_res]) if self.noise_std > 0 else None
        return self._apply_one(img, params, noise=noise)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._crop_one(img) for img in x], dim=0)

    def paired(self, x_a: torch.Tensor, x_b: torch.Tensor, same_noise: bool = False):
        if x_a.size(0) != x_b.size(0):
            raise ValueError('Paired augmentation expects batches with the same length.')
        out_a = []
        out_b = []
        for img_a, img_b in zip(x_a, x_b):
            _, h, w = img_a.shape
            params = self._sample_params(h, w)
            shared_noise = None
            if self.noise_std > 0 and same_noise:
                shared_noise = torch.randn(3, self.crop_res, self.crop_res, device=img_a.device, dtype=img_a.dtype)
            out_a.append(self._apply_one(img_a, params, noise=shared_noise))
            out_b.append(self._apply_one(img_b, params, noise=shared_noise))
        return torch.stack(out_a, dim=0), torch.stack(out_b, dim=0)


def get_augmentor(aug_mode: str = 'standard', crop_res: int = 224, noise_std: float = 0.0):
    if aug_mode == 'none':
        return IdentityAugmentor()
    if aug_mode == 'standard':
        return StandardAugmentor(crop_res=crop_res, noise_std=float(noise_std))
    raise ValueError(f'Unknown aug_mode: {aug_mode}')
