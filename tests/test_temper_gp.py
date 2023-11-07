import gpytorch
import pytest
import torch

from mockers.kernel import MockKernel
from src.gps import ExactGP
from src.temper import TemperGP


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
def test_temper_gp_mean(
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
    temper_gp = TemperGP(
        gp=gp,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    assert torch.allclose(temper_gp(x).mean, expected_mean)


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
                    [2.1850500106811523, 2.2330899238586426],
                    [2.2330899238586426, 7.536156177520752],
                ]
            ),
        ],
    ],
)
def test_temper_gp_expected_covariance_matrix(
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
    temper_gp = TemperGP(
        gp=gp,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    assert torch.allclose(temper_gp(x).covariance_matrix, expected_covariance_matrix)
