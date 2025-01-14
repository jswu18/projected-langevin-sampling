import gpytorch
import numpy as np
import pytest
import torch

from mockers.basis import MockBasis
from mockers.cost import MockCost
from mockers.kernel import MockKernel
from src.custom_types import GP_TYPE
from src.gaussian_process import SVGP, ExactGP
from src.projected_langevin_sampling import PLS
from src.temper import TemperGP, TemperPLS


@pytest.mark.parametrize(
    "gp,x_calibration,y_calibration",
    [
        [
            ExactGP(
                mean=gpytorch.means.ConstantMean(),
                kernel=MockKernel(),
                x=torch.tensor(
                    [
                        [1.1, 3.5, 3.5],
                        [1.3, 7.5, 1.5],
                        [2.5, 2.5, 0.5],
                        [1.0, 2.0, 3.0],
                        [1.5, 2.5, 3.5],
                    ]
                ),
                y=torch.tensor([3.1, 3.4, 7.8, 0.9, 1.1]),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            ),
            torch.tensor(
                [
                    [2.1, 2.5, 1.5],
                    [6.3, 7.3, 1.5],
                    [3.5, 2.8, 0.5],
                    [4.0, 2.1, 2.0],
                    [9.5, 2.3, 3.5],
                ]
            ),
            torch.tensor([8.1, 3.0, 2.8, 7.9, 8.1]),
        ],
        [
            SVGP(
                mean=gpytorch.means.ConstantMean(),
                kernel=MockKernel(),
                x_induce=torch.tensor(
                    [
                        [1.1, 3.5, 3.5],
                        [1.3, 7.5, 1.5],
                        [2.5, 2.5, 0.5],
                        [1.0, 2.0, 3.0],
                        [1.5, 2.5, 3.5],
                    ]
                ),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            ),
            torch.tensor(
                [
                    [2.1, 2.5, 1.5],
                    [6.3, 7.3, 1.5],
                    [3.5, 2.8, 0.5],
                    [4.0, 2.1, 2.0],
                    [9.5, 2.3, 3.5],
                ]
            ),
            torch.tensor([8.1, 3.0, 2.8, 7.9, 8.1]),
        ],
    ],
)
def test_temper_gp(
    gp: GP_TYPE,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
):
    gp.eval()
    temper_gp = TemperGP(
        gp=gp, x_calibration=x_calibration, y_calibration=y_calibration
    )
    assert isinstance(
        temper_gp(x_calibration),
        gpytorch.distributions.MultivariateNormal,
    )


@pytest.mark.parametrize(
    "gp,x_calibration,y_calibration,expected_scale",
    [
        [
            ExactGP(
                mean=gpytorch.means.ConstantMean(),
                kernel=MockKernel(),
                x=torch.tensor(
                    [
                        [1.1, 3.5, 3.5],
                        [1.3, 7.5, 1.5],
                        [2.5, 2.5, 0.5],
                        [1.0, 2.0, 3.0],
                        [1.5, 2.5, 3.5],
                    ]
                ),
                y=torch.tensor([3.1, 3.4, 7.8, 0.9, 1.1]),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            ),
            torch.tensor(
                [
                    [2.1, 2.5, 1.5],
                    [6.3, 7.3, 1.5],
                    [3.5, 2.8, 0.5],
                    [4.0, 2.1, 2.0],
                    [9.5, 2.3, 3.5],
                ]
            ),
            torch.tensor([8.1, 3.0, 2.8, 7.9, 8.1]),
            38.135372161865234,
        ],
        [
            SVGP(
                mean=gpytorch.means.ConstantMean(),
                kernel=MockKernel(),
                x_induce=torch.tensor(
                    [
                        [1.1, 3.5, 3.5],
                        [1.3, 7.5, 1.5],
                        [2.5, 2.5, 0.5],
                        [1.0, 2.0, 3.0],
                        [1.5, 2.5, 3.5],
                    ]
                ),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            ),
            torch.tensor(
                [
                    [2.1, 2.5, 1.5],
                    [6.3, 7.3, 1.5],
                    [3.5, 2.8, 0.5],
                    [4.0, 2.1, 2.0],
                    [9.5, 2.3, 3.5],
                ]
            ),
            torch.tensor([8.1, 3.0, 2.8, 7.9, 8.1]),
            3.3517842292785645,
        ],
    ],
)
def test_temper_gp_scale(
    gp: GP_TYPE,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    expected_scale: float,
):
    gp.eval()
    temper_gp = TemperGP(
        gp=gp, x_calibration=x_calibration, y_calibration=y_calibration
    )
    assert np.allclose(
        temper_gp.scale,
        expected_scale,
    )


@pytest.mark.parametrize(
    "particles,x_calibration,y_calibration",
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
            torch.tensor(
                [
                    [2.1, 2.5, 1.5],
                    [6.3, 7.3, 1.5],
                    [3.5, 2.8, 0.5],
                    [4.0, 2.1, 2.0],
                    [9.5, 2.3, 3.5],
                ]
            ),
            torch.tensor([8.1, 3.0, 2.8, 7.9, 8.1]),
        ],
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
            torch.tensor(
                [
                    [2.1, 2.5, 1.5],
                    [6.3, 7.3, 1.5],
                    [3.5, 2.8, 0.5],
                    [4.0, 2.1, 2.0],
                    [9.5, 2.3, 3.5],
                ]
            ),
            torch.tensor([8.1, 3.0, 2.8, 7.9, 8.1]),
        ],
    ],
)
def test_temper_pls(
    particles: torch.Tensor,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
):
    temper_pls = TemperPLS(
        pls=PLS(
            basis=MockBasis(),
            cost=MockCost(),
        ),
        particles=particles,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
        debug=True,
    )
    assert isinstance(
        temper_pls(x_calibration),
        gpytorch.distributions.MultivariateNormal,
    )


@pytest.mark.parametrize(
    "particles,x_calibration,y_calibration,expected_scale",
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
            torch.tensor(
                [
                    [2.1, 2.5, 1.5],
                    [6.3, 7.3, 1.5],
                    [3.5, 2.8, 0.5],
                    [4.0, 2.1, 2.0],
                    [9.5, 2.3, 3.5],
                ]
            ),
            torch.tensor([8.1, 3.0, 2.8, 7.9, 8.1]),
            84.18800354003906,
        ],
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
            torch.tensor(
                [
                    [2.1, 2.5, 1.5],
                    [6.3, 7.3, 1.5],
                    [3.5, 2.8, 0.5],
                    [4.0, 2.1, 2.0],
                    [9.5, 2.3, 3.5],
                ]
            ),
            torch.tensor([8.1, 3.0, 2.8, 7.9, 8.1]),
            84.18800354003906,
        ],
    ],
)
def test_temper_pls_scale(
    particles: torch.Tensor,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    expected_scale: float,
):
    temper_pls = TemperPLS(
        pls=PLS(
            basis=MockBasis(),
            cost=MockCost(),
        ),
        particles=particles,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
        debug=True,
    )
    assert np.allclose(
        temper_pls.scale,
        expected_scale,
    )
