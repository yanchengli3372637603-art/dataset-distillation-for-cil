from typing import Optional, Tuple

import torch
from torch import Tensor


class BaseDistilledDataset:
    """Ported from the official linear-gradient-matching repository.

    The color decorrelation path follows the official implementation:
    use the fixed 3x3 SVD-sqrt correlation matrix, normalize it by the
    maximum column L2 norm, and then right-multiply the flattened pixels.
    This version intentionally avoids an `einops` dependency so it can run
    in the original MACIL environment without extra installs.
    """

    optimizer: torch.optim.Optimizer
    syn_lr: Tensor
    res: int
    num_samples: int

    def __init__(self, device: torch.device):
        self.device = device
        self.color_correlation_svd_sqrt = torch.tensor(
            [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]],
            device=device,
            dtype=torch.float32,
        )
        self.max_norm_svd_sqrt = torch.max(
            torch.linalg.norm(self.color_correlation_svd_sqrt, dim=0)
        )
        self.color_mean = torch.tensor([0.48, 0.46, 0.41], device=device, dtype=torch.float32)

    def get_data(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def log_images(self, step: Optional[int] = None):
        raise NotImplementedError

    def upkeep(self, step: Optional[int] = None):
        return

    def get_save_dict(self):
        return {}

    def load_from_dict(self, load_dict: dict):
        return

    def linear_decorrelate_color(self, im: Tensor):
        b, c, h, w = im.shape
        flat = im.permute(0, 2, 3, 1).reshape(b * h * w, c)
        color_correlation_normalized = self.color_correlation_svd_sqrt / self.max_norm_svd_sqrt
        flat = flat @ color_correlation_normalized.T
        im = flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return im
