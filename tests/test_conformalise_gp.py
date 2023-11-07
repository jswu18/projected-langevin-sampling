import gpytorch
import pytest
import torch

from mockers.kernel import MockKernel
from src.conformalise import ConformaliseGP
from src.gps import ExactGP


@pytest.mark.parametrize(
    "x_train,y_train,x_calibration,y_calibration,x,expected_mean",
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
            torch.tensor([0.1, 2.3, 3.1, 2.1, 3.3]),
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor([2.1, 3.3]),
            torch.tensor(
                [
                    [3.2, 4.2, 4.0],
                    [5.1, 2.1, 9.5],
                ]
            ),
            torch.tensor([4.2897491455078125, 6.886444091796875]),
        ],
    ],
)
def test_conformalise_gp_mean(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    x: torch.Tensor,
    expected_mean: torch.Tensor,
):
    gp = ExactGP(
        mean=gpytorch.means.ConstantMean(),
        kernel=MockKernel(),
        x=x_train,
        y=y_train,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    )
    gp.eval()
    conformal_gp = ConformaliseGP(
        gp=gp,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    assert torch.allclose(conformal_gp(x).mean, expected_mean)


@pytest.mark.parametrize(
    "x_train,y_train,x_calibration,y_calibration,x,expected_covariance_matrix",
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
            torch.tensor([0.1, 2.3, 3.1, 2.1, 3.3]),
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor([2.1, 3.3]),
            torch.tensor(
                [
                    [3.2, 4.2, 4.0],
                    [5.1, 2.1, 9.5],
                ]
            ),
            torch.tensor(
                [
                    [1.2794084548950195, 0],
                    [0, 2.240105628967285],
                ]
            ),
        ],
    ],
)
def test_conformalise_gp_expected_covariance_matrix(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    x: torch.Tensor,
    expected_covariance_matrix: torch.Tensor,
):
    gp = ExactGP(
        mean=gpytorch.means.ConstantMean(),
        kernel=MockKernel(),
        x=x_train,
        y=y_train,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    )
    gp.eval()
    conformal_gp = ConformaliseGP(
        gp=gp,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    assert torch.allclose(conformal_gp(x).covariance_matrix, expected_covariance_matrix)


@pytest.mark.parametrize(
    "x_train,y_train,x_calibration,y_calibration,x,coverage,expected_interval_width",
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
            torch.tensor([0.1, 2.3, 3.1, 2.1, 3.3]),
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor([2.1, 3.3]),
            torch.tensor(
                [
                    [3.2, 4.2, 4.0],
                    [5.1, 2.1, 9.5],
                ]
            ),
            0.95,
            4.863025665283203,
        ],
    ],
)
def test_conformalise_gp_average_interval_width(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    x: torch.Tensor,
    coverage: float,
    expected_interval_width: float,
):
    gp = ExactGP(
        mean=gpytorch.means.ConstantMean(),
        kernel=MockKernel(),
        x=x_train,
        y=y_train,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    )
    gp.eval()
    conformal_gp = ConformaliseGP(
        gp=gp,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    assert (
        conformal_gp.calculate_average_interval_width(x=x, coverage=coverage).item()
        == expected_interval_width
    )
