import pytest
import torch

from mockers.kernel import MockKernel, MockPLSKernel
from src.projected_langevin_sampling.basis import InducingPointBasis, OrthonormalBasis
from src.utils import set_seed


@pytest.mark.parametrize(
    "x_induce,x_train,eigenvalue_threshold,approximation_dimension",
    [
        [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            0.0,
            2,
        ],
        [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            1.0,
            1,
        ],
    ],
)
def test_onb_approximation_dimension(
    x_induce: torch.Tensor,
    x_train: torch.Tensor,
    eigenvalue_threshold: float,
    approximation_dimension: int,
):
    assert (
        OrthonormalBasis(
            kernel=MockPLSKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            x_train=x_train,
            eigenvalue_threshold=eigenvalue_threshold,
        ).approximation_dimension
        == approximation_dimension
    )


@pytest.mark.parametrize(
    "x_induce,y_induce,x_train,approximation_dimension",
    [
        [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor([2.1, 3.3]),
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            2,
        ],
    ],
)
def test_ipb_approximation_dimension(
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    approximation_dimension: int,
):
    assert (
        InducingPointBasis(
            kernel=MockPLSKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
        ).approximation_dimension
        == approximation_dimension
    )


@pytest.mark.parametrize(
    "x_induce,x_train,eigenvalue_threshold,number_of_particles,initialised_particles",
    [
        [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            0.0,
            3,
            torch.tensor(
                [
                    [1.5409960746765137, -0.293428897857666, -2.1787893772125244],
                    [0.5684312582015991, -1.0845223665237427, -1.3985954523086548],
                ]
            ),
        ],
        [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            1.0,
            3,
            torch.tensor(
                [[1.5409960746765137, -0.293428897857666, -2.1787893772125244]]
            ),
        ],
    ],
)
def test_onb_initialised_particles(
    x_induce: torch.Tensor,
    x_train: torch.Tensor,
    eigenvalue_threshold: float,
    number_of_particles: int,
    initialised_particles: torch.Tensor,
):
    assert torch.allclose(
        OrthonormalBasis(
            kernel=MockPLSKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            x_train=x_train,
            eigenvalue_threshold=eigenvalue_threshold,
        ).initialise_particles(number_of_particles=number_of_particles, seed=0),
        initialised_particles,
    )


@pytest.mark.parametrize(
    "x_induce,y_induce,x_train,number_of_particles,noise_only,initialised_particles",
    [
        [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor([2.1, 3.3]),
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            2,
            True,
            torch.tensor(
                [
                    [1.5409960746765137, -0.293428897857666],
                    [-2.1787893772125244, 0.5684312582015991],
                ]
            ),
        ],
        [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor([2.1, 3.3]),
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            2,
            False,
            torch.tensor(
                [
                    [3.640995979309082, 1.8065710067749023],
                    [1.1212105751037598, 3.8684310913085938],
                ]
            ),
        ],
    ],
)
def test_ipb_initialised_particles(
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    number_of_particles: int,
    noise_only: bool,
    initialised_particles: torch.Tensor,
):
    assert torch.allclose(
        InducingPointBasis(
            kernel=MockPLSKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
        ).initialise_particles(
            number_of_particles=number_of_particles, seed=0, noise_only=noise_only
        ),
        initialised_particles,
    )


@pytest.mark.parametrize(
    "x_induce,x_train,eigenvalue_threshold,particles,untransformed_train_prediction_samples",
    [
        [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            0.0,
            torch.tensor(
                [
                    [1.5409960746765137, -0.293428897857666, -2.1787893772125244],
                    [0.5684312582015991, -1.0845223665237427, -1.3985954523086548],
                ]
            ),
            torch.tensor(
                [
                    [2.8012976646, -5.3903346062, -6.9202804565],
                    [6.0996646881, -6.6724033356, -11.9823102951],
                    [5.1316790581, -3.4285004139, -8.7493305206],
                    [1.8773055077, -4.0070199966, -4.8781495094],
                    [2.7915179729, -4.9768695831, -6.6556425095],
                ]
            ),
        ],
    ],
)
def test_onb_predict_untransformed_samples(
    x_induce: torch.Tensor,
    x_train: torch.Tensor,
    eigenvalue_threshold: float,
    particles: torch.Tensor,
    untransformed_train_prediction_samples: torch.Tensor,
):
    assert torch.allclose(
        OrthonormalBasis(
            kernel=MockPLSKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            x_train=x_train,
            eigenvalue_threshold=eigenvalue_threshold,
        ).calculate_untransformed_train_prediction_samples(particles),
        untransformed_train_prediction_samples,
    )


@pytest.mark.parametrize(
    "x_induce,y_induce,x_train,particles,untransformed_train_prediction_samples",
    [
        [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [1.5, 2.5, 3.5],
                ]
            ),
            torch.tensor([2.1, 3.3]),
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
                    [1.5409960746765137, -0.293428897857666, -2.1787893772125244],
                    [0.5684312582015991, -1.0845223665237427, -1.3985954523086548],
                ]
            ),
            torch.tensor(
                [
                    [1.2656511068, -0.8267806172, -2.1464431286],
                    [-6.1349906921, -5.1450047493, 4.8272337914],
                    [-8.9970912933, -5.7714910507, 8.1600494385],
                    [1.5409765244, -0.2934455872, -2.1787719727],
                    [0.5684509277, -1.0845184326, -1.3986053467],
                ]
            ),
        ],
    ],
)
def test_ipb_predict_untransformed_samples(
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    particles: torch.Tensor,
    untransformed_train_prediction_samples: torch.Tensor,
):
    assert torch.allclose(
        InducingPointBasis(
            kernel=MockPLSKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
        ).calculate_untransformed_train_prediction_samples(particles),
        untransformed_train_prediction_samples,
    )
