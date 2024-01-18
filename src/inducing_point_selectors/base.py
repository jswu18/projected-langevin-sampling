from abc import ABC, abstractmethod
from typing import Tuple

import gpytorch
import torch


class InducingPointSelector(ABC):
    @abstractmethod
    def compute_induce_data(
        self,
        x: torch.Tensor,
        m: int,
        kernel: gpytorch.kernels.Kernel,
        **params,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inducing points and the corresponding inducing indices.
        """
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        m: int,
        kernel: gpytorch.kernels.Kernel,
        **params,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compute_induce_data(
            x=x,
            m=m,
            kernel=kernel,
            **params,
        )
