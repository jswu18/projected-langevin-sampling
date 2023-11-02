from typing import Tuple

import gpytorch
import torch

from src.induce_data_selectors.base import InduceDataSelector


class RandomInduceDataSelector(InduceDataSelector):
    def compute_induce_data(
        self,
        x: torch.Tensor,
        m: int,
        kernel: gpytorch.kernels.Kernel = None,
        **params,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randperm(x.shape[0])[:m]
        return x[indices, ...], indices
