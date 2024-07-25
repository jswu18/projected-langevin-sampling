from typing import Tuple

import gpytorch
import torch

from src.inducing_point_selectors.base import InducingPointSelector


class RandomInducingPointSelector(InducingPointSelector):
    def compute_induce_data(
        self,
        x: torch.Tensor,
        m: int,
        kernel: gpytorch.kernels.Kernel = None,
        **params,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(x.shape[0])[:m]
        return x[indices, ...], indices
