import gpytorch
import numpy as np
import pytest
import torch

from mockers.basis import MockBasis
from mockers.cost import MockCost
from mockers.kernel import MockKernel
from src.conformalise import ConformaliseGP, ConformalisePLS
from src.conformalise.base import ConformalPrediction
from src.custom_types import GP_TYPE
from src.gaussian_process import SVGP, ExactGP
from src.projected_langevin_sampling import PLS


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
    conformal_gp = ConformaliseGP(
        gp=gp, x_calibration=x_calibration, y_calibration=y_calibration
    )
    assert isinstance(
        conformal_gp(x_calibration, coverage=0.95),
        ConformalPrediction,
    )


@pytest.mark.parametrize(
    "gp,x_calibration,y_calibration,expected_average_interval_width",
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
            26.48227882385254,
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
            28.007099151611328,
        ],
    ],
)
def test_temper_gp_scale(
    gp: GP_TYPE,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    expected_average_interval_width: float,
):
    gp.eval()
    conformal_gp = ConformaliseGP(
        gp=gp, x_calibration=x_calibration, y_calibration=y_calibration
    )
    assert np.allclose(
        conformal_gp.calculate_average_interval_width(x=x_calibration, coverage=0.95),
        expected_average_interval_width,
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
    conformal_pls = ConformalisePLS(
        pls=PLS(
            basis=MockBasis(),
            cost=MockCost(),
        ),
        particles=particles,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    assert isinstance(
        conformal_pls(x_calibration, coverage=0.95),
        ConformalPrediction,
    )


@pytest.mark.parametrize(
    "particles,x_calibration,y_calibration,expected_average_interval_width",
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
            327.94561767578125,
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
            327.94561767578125,
        ],
    ],
)
def test_temper_pls_scale(
    particles: torch.Tensor,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    expected_average_interval_width: float,
):
    conformal_pls = ConformalisePLS(
        pls=PLS(
            basis=MockBasis(),
            cost=MockCost(),
        ),
        particles=particles,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    assert np.allclose(
        conformal_pls.calculate_average_interval_width(
            x=x_calibration,
            coverage=0.95,
        ),
        expected_average_interval_width,
    )
