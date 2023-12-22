import pytest
import torch

from mockers.kernel import MockGradientFlowKernel, MockKernel
from src.conformalise import ConformaliseGradientFlow
from src.gradient_flows.regression import GradientFlowRegression
from src.utils import set_seed


@pytest.mark.parametrize(
    "x_train,y_train,x_induce,y_induce,x_calibration,y_calibration,jitter,observation_noise,particles,x,expected_mean",
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
                    [1.4, 4.0, 3.0],
                    [2.5, 2.2, 1.5],
                ]
            ),
            torch.tensor([1.1, 1.3]),
            0.0,
            1.0,
            torch.tensor(
                [
                    [-4.177099609326439, 7.969579237856498, 10.277535644199697],
                    [0.1856488604600999, -0.35420350057036537, -0.45677933473829735],
                ]
            ),
            torch.tensor(
                [
                    [3.2, 4.2, 4.0],
                    [5.1, 2.1, 9.5],
                ]
            ),
            torch.tensor([-45.917715538988084, -6.5844544039101045]),
        ],
    ],
)
def test_conformalise_pwgf_mean(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    jitter: float,
    observation_noise: float,
    particles: torch.Tensor,
    x: torch.Tensor,
    expected_mean: torch.Tensor,
):
    kernel = MockGradientFlowKernel(
        base_kernel=MockKernel(),
        approximation_samples=x_train.double(),
    )
    pwgf = GradientFlowRegression(
        number_of_particles=particles.shape[1],
        kernel=kernel,
        x_induce=x_induce.double(),
        y_induce=y_induce.double(),
        x_train=x_train.double(),
        y_train=y_train.double(),
        jitter=jitter,
        observation_noise=observation_noise,
    )
    pwgf.particles = particles.double()
    set_seed(0)
    conformal_gradient_flow = ConformaliseGradientFlow(
        gradient_flow=pwgf,
        x_calibration=x_calibration.double(),
        y_calibration=y_calibration.double(),
    )
    assert torch.allclose(
        conformal_gradient_flow(x.double()).mean, expected_mean.double()
    )


@pytest.mark.parametrize(
    """
    x_train,y_train,x_induce,y_induce,x_calibration,y_calibration,
    jitter,observation_noise,particles,x,expected_covariance_matrix
    """,
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
                    [1.4, 4.0, 3.0],
                    [2.5, 2.2, 1.5],
                ]
            ),
            torch.tensor([1.1, 1.3]),
            0.0,
            1.0,
            torch.tensor(
                [
                    [-4.177099609326439, 7.969579237856498, 10.277535644199697],
                    [0.1856488604600999, -0.35420350057036537, -0.45677933473829735],
                ]
            ),
            torch.tensor(
                [
                    [3.2, 4.2, 4.0],
                    [5.1, 2.1, 9.5],
                ]
            ),
            torch.tensor(
                [
                    [29.132120605438356, 0],
                    [0, 3.2602388858795166],
                ]
            ),
        ],
    ],
)
def test_conformalise_pwgf_covariance_matrix(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    jitter: float,
    observation_noise: float,
    particles: torch.Tensor,
    x: torch.Tensor,
    expected_covariance_matrix: torch.Tensor,
):
    kernel = MockGradientFlowKernel(
        base_kernel=MockKernel(),
        approximation_samples=x_train.double(),
    )
    pwgf = GradientFlowRegression(
        number_of_particles=particles.shape[1],
        kernel=kernel,
        x_induce=x_induce.double(),
        y_induce=y_induce.double(),
        x_train=x_train.double(),
        y_train=y_train.double(),
        jitter=jitter,
        observation_noise=observation_noise,
    )
    pwgf.particles = particles.double()
    conformal_gradient_flow = ConformaliseGradientFlow(
        gradient_flow=pwgf,
        x_calibration=x_calibration.double(),
        y_calibration=y_calibration.double(),
    )
    set_seed(0)
    assert torch.allclose(
        conformal_gradient_flow(x.double()).covariance_matrix,
        expected_covariance_matrix.double(),
    )


@pytest.mark.parametrize(
    """
    x_train,y_train,x_induce,y_induce,x_calibration,y_calibration,
    jitter,observation_noise,particles,x,coverage,expected_interval_width
    """,
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
                    [1.4, 4.0, 3.0],
                    [2.5, 2.2, 1.5],
                ]
            ),
            torch.tensor([1.1, 1.3]),
            0.0,
            1.0,
            torch.tensor(
                [
                    [-4.177099609326439, 7.969579237856498, 10.277535644199697],
                    [0.1856488604600999, -0.35420350057036537, -0.45677933473829735],
                ]
            ),
            torch.tensor(
                [
                    [3.2, 4.2, 4.0],
                    [5.1, 2.1, 9.5],
                ]
            ),
            0.95,
            13.189324758638877,
        ],
    ],
)
def test_conformalise_pwgf_average_interval_width(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_calibration: torch.Tensor,
    y_calibration: torch.Tensor,
    jitter: float,
    observation_noise: float,
    particles: torch.Tensor,
    x: torch.Tensor,
    coverage: float,
    expected_interval_width: float,
):
    set_seed(0)
    kernel = MockGradientFlowKernel(
        base_kernel=MockKernel(),
        approximation_samples=x_train.double(),
    )
    pwgf = GradientFlowRegression(
        number_of_particles=particles.shape[1],
        kernel=kernel,
        x_induce=x_induce.double(),
        y_induce=y_induce.double(),
        x_train=x_train.double(),
        y_train=y_train.double(),
        jitter=jitter,
        observation_noise=observation_noise,
    )
    conformal_gradient_flow = ConformaliseGradientFlow(
        gradient_flow=pwgf,
        x_calibration=x_calibration.double(),
        y_calibration=y_calibration.double(),
    )
    assert (
        conformal_gradient_flow.calculate_average_interval_width(
            x=x.double(), coverage=coverage
        ).item()
        == expected_interval_width
    )
