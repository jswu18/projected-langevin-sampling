import gpytorch
import linear_operator
import torch

from src.kernels import PLSKernel


class MockKernel(gpytorch.kernels.Kernel):
    """
    A mock kernel used for testing that computes the inner product between the inputs.
    """

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        last_dim_is_batch: bool = False,
        diag: bool = False,
        **params,
    ) -> linear_operator.operators.LinearOperator:
        if diag:
            return (x1 * x2).sum(-1)
        return x1 @ x2.transpose(-1, -2)


class MockPLSKernel(PLSKernel):
    """
    A mock kernel used for testing that computes the inner product between the inputs.
    """

    base_kernel = MockKernel()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        last_dim_is_batch: bool = False,
        diag: bool = False,
        **params,
    ) -> linear_operator.operators.LinearOperator:
        if diag:
            return (x1 * x2).sum(-1)
        return x1 @ x2.transpose(-1, -2)
