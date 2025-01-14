import gpytorch
import pytest
import torch

from mockers.kernel import MockKernel
from src.gaussian_process import SVGP, ExactGP


@pytest.mark.parametrize(
    "x,y",
    [
        [
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor([3.1, 3.4, 7.8, 0.9, 1.1]),
        ],
    ],
)
def test_exact_gp(
    x: torch.Tensor,
    y: torch.Tensor,
):
    assert isinstance(
        ExactGP(
            mean=gpytorch.means.ConstantMean(),
            kernel=MockKernel(),
            x=x,
            y=y,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )(x),
        gpytorch.distributions.MultivariateNormal,
    )


@pytest.mark.parametrize(
    "x_induce",
    [
        torch.tensor(
            [
                [1.1, 3.5, 3.5],
                [1.3, 7.5, 1.5],
                [2.5, 2.5, 0.5],
                [1.0, 2.0, 3.0],
                [1.5, 2.5, 3.5],
            ]
        ),
    ],
)
def test_svgp(
    x_induce: torch.Tensor,
):
    assert isinstance(
        SVGP(
            mean=gpytorch.means.ConstantMean(),
            kernel=MockKernel(),
            x_induce=x_induce,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        )(x_induce),
        gpytorch.distributions.MultivariateNormal,
    )
