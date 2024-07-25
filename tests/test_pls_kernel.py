import pytest
import torch

from mockers.kernel import MockKernel
from src.kernels import PLSKernel


@pytest.mark.parametrize(
    "x1,x2,z,gram",
    [
        [
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([[1.5, 2.5, 3.5]]),
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                ]
            ),
            torch.tensor(355.60004),
        ],
        [
            torch.tensor(
                [
                    [1.0, 3.0],
                    [3.0, 5.0],
                ]
            ),
            torch.tensor([[1.5, 3.5]]),
            torch.tensor(
                [
                    [1.1, 3.5],
                    [1.3, 7.5],
                    [2.5, 2.5],
                ]
            ),
            torch.tensor([[319.13333], [568.8667]]),
        ],
    ],
)
def test_pls_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    z: torch.Tensor,
    gram: torch.Tensor,
):
    kernel = PLSKernel(
        base_kernel=MockKernel(),
        approximation_samples=z,
    )
    assert torch.allclose(torch.tensor(kernel(x1, x2).numpy()), gram)
