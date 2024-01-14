import gpytorch
import torch

from src.kernels.base import GradientFlowBaseKernel


class GradientFlowONBKernel(GradientFlowBaseKernel):
    """
    Decorates a base kernel with a gradient flow kernel such that
    K(x1, x2) = 1 / N * sum_{i=1}^N K_b(x1, z_i) * K_b(x2, z_i)
    where K_b is the base kernel and z_i are the approximation samples.
    """

    is_stationary = False

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        approximation_samples: torch.Tensor,
        **kwargs,
    ):
        super().__init__(
            base_kernel=base_kernel,
            approximation_samples=approximation_samples,
            **kwargs,
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        last_dim_is_batch: bool = False,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        combined_approximation_samples = torch.cat(
            [self.approximation_samples, x1, x2], dim=0
        ).unique(dim=0)
        combined_approximation_samples = torch.sort(
            combined_approximation_samples, dim=0
        )[0]
        number_of_approximation_samples = combined_approximation_samples.shape[0]
        gram_x1_sample = self.base_kernel.forward(
            x1=x1,
            x2=combined_approximation_samples,
            last_dim_is_batch=last_dim_is_batch,
        )
        gram_x2_sample = self.base_kernel.forward(
            x1=x2,
            x2=combined_approximation_samples,
            last_dim_is_batch=last_dim_is_batch,
        )
        if diag:
            return torch.mul(
                torch.div(1, number_of_approximation_samples),
                (gram_x1_sample @ gram_x2_sample.T),
            ).diag()
        return torch.mul(
            torch.div(1, number_of_approximation_samples),
            (gram_x1_sample @ gram_x2_sample.T),
        )
