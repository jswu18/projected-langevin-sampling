import numpy as np
import pytest
import torch

from mockers.cost import MockCost
from mockers.kernel import MockKernel, MockProjectedLangevinSamplingKernel
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
            kernel=MockProjectedLangevinSamplingKernel(
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
            kernel=MockProjectedLangevinSamplingKernel(
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
            kernel=MockProjectedLangevinSamplingKernel(
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
            kernel=MockProjectedLangevinSamplingKernel(
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
def test_onb_calculate_untransformed_train_prediction_samples(
    x_induce: torch.Tensor,
    x_train: torch.Tensor,
    eigenvalue_threshold: float,
    particles: torch.Tensor,
    untransformed_train_prediction_samples: torch.Tensor,
):
    assert torch.allclose(
        OrthonormalBasis(
            kernel=MockProjectedLangevinSamplingKernel(
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
def test_ipb_calculate_untransformed_train_prediction_samples(
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    particles: torch.Tensor,
    untransformed_train_prediction_samples: torch.Tensor,
):
    assert torch.allclose(
        InducingPointBasis(
            kernel=MockProjectedLangevinSamplingKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
        ).calculate_untransformed_train_prediction_samples(particles),
        untransformed_train_prediction_samples,
    )


@pytest.mark.parametrize(
    "x_induce,x_train,eigenvalue_threshold,particles,untransformed_train_prediction_samples,expected_energy_potential",
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
            56.62522888183594,
        ],
    ],
)
def test_onb_calculate_energy_potential(
    x_induce: torch.Tensor,
    x_train: torch.Tensor,
    eigenvalue_threshold: float,
    particles: torch.Tensor,
    untransformed_train_prediction_samples: torch.Tensor,
    expected_energy_potential: float,
):
    assert np.allclose(
        OrthonormalBasis(
            kernel=MockProjectedLangevinSamplingKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            x_train=x_train,
            eigenvalue_threshold=eigenvalue_threshold,
        ).calculate_energy_potential(
            particles=particles,
            cost=MockCost().calculate_cost(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            ),
        ),
        expected_energy_potential,
    )


@pytest.mark.parametrize(
    "x_induce,y_induce,x_train,particles,untransformed_train_prediction_samples,expected_energy_potential",
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
            275.2294006347656,
        ],
    ],
)
def test_ipb_calculate_energy_potential(
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    particles: torch.Tensor,
    untransformed_train_prediction_samples: torch.Tensor,
    expected_energy_potential: float,
):
    assert np.allclose(
        InducingPointBasis(
            kernel=MockProjectedLangevinSamplingKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
        ).calculate_energy_potential(
            particles=particles,
            cost=MockCost().calculate_cost(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            ),
        ),
        expected_energy_potential,
    )


@pytest.mark.parametrize(
    "x_induce,x_train,eigenvalue_threshold,particles,x,additional_predictive_noise_distribution,expected_predictive_noise",
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
                    [3.0, 2.0, 3.2],
                    [1.5, 6.5, 1.5],
                ]
            ),
            None,
            torch.tensor(
                [
                    [0.0851, -0.1569, -0.2067],
                    [3.1662, 4.6236, -1.2954],
                    [1.3697, 1.4171, 0.7368],
                    [3.9759, 6.4164, -2.9854],
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
            0.0,
            torch.tensor(
                [
                    [1.5409960746765137, -0.293428897857666, -2.1787893772125244],
                    [0.5684312582015991, -1.0845223665237427, -1.3985954523086548],
                ]
            ),
            torch.tensor(
                [
                    [3.0, 2.0, 3.2],
                    [1.5, 6.5, 1.5],
                ]
            ),
            torch.distributions.studentT.StudentT(
                loc=0.0,
                scale=1.0,
                df=3,
            ),
            torch.tensor(
                [
                    [-0.8059, 0.6046, -1.4332],
                    [3.3688, 4.0346, -0.9600],
                    [-0.4252, -0.7840, 1.1648],
                    [5.4293, 8.3857, -1.9861],
                ]
            ),
        ],
    ],
)
def test_onb_sample_predictive_noise(
    x_induce: torch.Tensor,
    x_train: torch.Tensor,
    eigenvalue_threshold: float,
    particles: torch.Tensor,
    x: torch.Tensor,
    additional_predictive_noise_distribution: torch.distributions.Distribution | None,
    expected_predictive_noise: torch.Tensor,
):
    set_seed(0)
    assert torch.allclose(
        OrthonormalBasis(
            kernel=MockProjectedLangevinSamplingKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            x_train=x_train,
            eigenvalue_threshold=eigenvalue_threshold,
            additional_predictive_noise_distribution=additional_predictive_noise_distribution,
        ).sample_predictive_noise(
            particles=particles,
            x=x,
        ),
        expected_predictive_noise,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "x_induce,y_induce,x_train,particles,x,additional_predictive_noise_distribution,expected_predictive_noise",
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
                    [3.0, 2.0, 3.2],
                    [1.5, 6.5, 1.5],
                ]
            ),
            None,
            torch.tensor(
                [
                    [1.4442, 3.7593, -0.4158],
                    [1.9489, 4.2264, -0.9286],
                    [3.0377, 3.1129, -3.4442],
                    [1.4840, 1.3103, 0.6729],
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
            torch.tensor(
                [
                    [1.5409960746765137, -0.293428897857666, -2.1787893772125244],
                    [0.5684312582015991, -1.0845223665237427, -1.3985954523086548],
                ]
            ),
            torch.tensor(
                [
                    [3.0, 2.0, 3.2],
                    [1.5, 6.5, 1.5],
                ]
            ),
            torch.distributions.studentT.StudentT(
                loc=0.0,
                scale=1.0,
                df=3,
            ),
            torch.tensor(
                [
                    [0.5531, 4.5208, -1.6422],
                    [2.1516, 3.6374, -0.5932],
                    [1.2428, 0.9118, -3.0163],
                    [2.9374, 3.2796, 1.6723],
                ]
            ),
        ],
    ],
)
def test_ipb_sample_predictive_noise(
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    particles: torch.Tensor,
    x: torch.Tensor,
    additional_predictive_noise_distribution: torch.distributions.Distribution | None,
    expected_predictive_noise: torch.Tensor,
):
    set_seed(0)
    assert torch.allclose(
        InducingPointBasis(
            kernel=MockProjectedLangevinSamplingKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            additional_predictive_noise_distribution=additional_predictive_noise_distribution,
        ).sample_predictive_noise(
            particles=particles,
            x=x,
        ),
        expected_predictive_noise,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "x_induce,x_train,eigenvalue_threshold,particles,x,noise,expected_untransformed_samples",
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
                    [3.0, 2.0, 3.2],
                    [1.5, 6.5, 1.5],
                ]
            ),
            torch.tensor(
                [
                    [0.0851, -0.1569, -0.2067],
                    [3.1662, 4.6236, -1.2954],
                    [1.3697, 1.4171, 0.7368],
                    [3.9759, 6.4164, -2.9854],
                ]
            ),
            torch.tensor(
                [
                    [-8.1948, -24.4906, -2.6199],
                    [-6.7366, -23.3037, -7.1726],
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
            0.0,
            torch.tensor(
                [
                    [1.5409960746765137, -0.293428897857666, -2.1787893772125244],
                    [0.5684312582015991, -1.0845223665237427, -1.3985954523086548],
                ]
            ),
            torch.tensor(
                [
                    [3.0, 2.0, 3.2],
                    [1.5, 6.5, 1.5],
                ]
            ),
            None,
            torch.tensor(
                [
                    [-10.3824, -33.0209, -9.7953],
                    [-18.9473, -35.1754, -16.7981],
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
    x: torch.Tensor,
    noise: torch.Tensor | None,
    expected_untransformed_samples: torch.Tensor,
):
    set_seed(1)
    assert torch.allclose(
        OrthonormalBasis(
            kernel=MockProjectedLangevinSamplingKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            x_train=x_train,
            eigenvalue_threshold=eigenvalue_threshold,
        ).predict_untransformed_samples(particles=particles, x=x, noise=noise),
        expected_untransformed_samples,
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "x_induce,y_induce,x_train,particles,x,noise,expected_untransformed_samples",
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
                    [3.0, 2.0, 3.2],
                    [1.5, 6.5, 1.5],
                ]
            ),
            torch.tensor(
                [
                    [1.4442, 3.7593, -0.4158],
                    [1.9489, 4.2264, -0.9286],
                    [3.0377, 3.1129, -3.4442],
                    [1.4840, 1.3103, 0.6729],
                ]
            ),
            torch.tensor(
                [
                    [-4.4373, -3.6672, 2.9305],
                    [-7.8718, -6.6582, 8.8616],
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
            torch.tensor(
                [
                    [1.5409960746765137, -0.293428897857666, -2.1787893772125244],
                    [0.5684312582015991, -1.0845223665237427, -1.3985954523086548],
                ]
            ),
            torch.tensor(
                [
                    [3.0, 2.0, 3.2],
                    [1.5, 6.5, 1.5],
                ]
            ),
            None,
            torch.tensor(
                [
                    [-6.0771, -4.0037, 2.8162],
                    [-0.3138, -5.1959, 9.1097],
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
    x: torch.Tensor,
    noise: torch.Tensor,
    expected_untransformed_samples: torch.Tensor,
):
    set_seed(1)
    assert torch.allclose(
        InducingPointBasis(
            kernel=MockProjectedLangevinSamplingKernel(
                base_kernel=MockKernel(),
                approximation_samples=x_induce,
            ),
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
        ).predict_untransformed_samples(
            particles=particles,
            x=x,
            noise=noise,
        ),
        expected_untransformed_samples,
        rtol=1e-3,
    )
