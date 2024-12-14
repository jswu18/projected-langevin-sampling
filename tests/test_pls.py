import pytest
import torch

from mockers.basis import MockBasis
from mockers.cost import MockCost
from src.projected_langevin_sampling import PLS
from src.utils import set_seed


@pytest.mark.parametrize(
    "number_of_particles,seed,particles",
    [
        [
            3,
            0,
            torch.tensor(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ),
        ],
    ],
)
def test_initialise_particles(
    number_of_particles: int,
    seed: int,
    particles: torch.Tensor,
):
    pls = PLS(
        basis=MockBasis(),
        cost=MockCost(),
    )
    assert torch.allclose(
        pls.initialise_particles(number_of_particles=number_of_particles, seed=seed),
        particles,
    )


@pytest.mark.parametrize(
    "number_of_particles,seed,observation_noise",
    [
        [
            3,
            0,
            torch.tensor([0.0, 0.0, 0.0]),
        ],
    ],
)
def test_sample_observation_noise(
    number_of_particles: int,
    seed: int,
    observation_noise: torch.Tensor,
):
    pls = PLS(
        basis=MockBasis(),
        cost=MockCost(),
    )
    assert torch.allclose(
        pls.sample_observation_noise(
            number_of_particles=number_of_particles, seed=seed
        ),
        observation_noise,
    )


@pytest.mark.parametrize(
    "seed,particles,step_size,update",
    [
        [
            0,
            torch.tensor(
                [
                    [0.1, 0.0, 2.0],
                    [0.9, 0.0, 0.1],
                    [0.1, 0.5, 4.0],
                    [0.3, 0.0, 0.4],
                    [3.0, 0.0, 4.2],
                    [0.0, 2.0, 0.0],
                    [1.0, 0.1, 5.0],
                    [0.6, 0.4, 2.0],
                    [3.0, 0.2, 0.0],
                    [0.8, 0.9, 2.0],
                ]
            ),
            0.1,
            torch.tensor(
                [
                    [1.1000, 1.0000, 3.0000],
                    [1.9000, 1.0000, 1.1000],
                    [1.1000, 1.5000, 5.0000],
                    [1.3000, 1.0000, 1.4000],
                    [4.0000, 1.0000, 5.2000],
                    [1.0000, 3.0000, 1.0000],
                    [2.0000, 1.1000, 6.0000],
                    [1.6000, 1.4000, 3.0000],
                    [4.0000, 1.2000, 1.0000],
                    [1.8000, 1.9000, 3.0000],
                ]
            ),
        ],
    ],
)
def test_calculate_update(
    seed: int,
    particles: torch.Tensor,
    step_size: float,
    update: torch.Tensor,
):
    pls = PLS(
        basis=MockBasis(),
        cost=MockCost(),
    )
    set_seed(seed)
    calculated_update = pls.calculate_particle_update(
        particles=particles,
        step_size=torch.tensor(step_size),
    ).detach()
    assert torch.allclose(calculated_update, update)


@pytest.mark.parametrize(
    "x,seed,predict_noise",
    [
        [
            torch.tensor(
                [
                    [3.0, 12.0, 3.0],
                    [2.5, 2.5, 3.3],
                ]
            ),
            0,
            torch.tensor([[180.0], [83.0]]),
        ],
    ],
)
def test_sample_predictive_noise(
    x: torch.Tensor,
    seed: int,
    predict_noise: torch.Tensor,
):
    pls = PLS(
        basis=MockBasis(),
        cost=MockCost(),
    )
    set_seed(seed)
    particles = pls.initialise_particles(number_of_particles=1, seed=seed)
    sampled_predict_noise = pls.sample_predictive_noise(
        particles=particles,
        x=x,
    ).detach()
    assert torch.allclose(sampled_predict_noise, predict_noise)


@pytest.mark.parametrize(
    "seed,expected_cost",
    [
        [
            0,
            torch.tensor([[1.0]]),
        ],
    ],
)
def test_calculate_cost(
    seed: int,
    expected_cost: torch.Tensor,
):
    pls = PLS(
        basis=MockBasis(),
        cost=MockCost(),
    )
    set_seed(seed)
    particles = pls.initialise_particles(number_of_particles=1, seed=seed)
    cost = pls.calculate_cost(
        particles=particles,
    ).detach()
    assert torch.allclose(cost, expected_cost)


@pytest.mark.parametrize(
    "x,seed,particles,prediction",
    [
        [
            torch.tensor(
                [
                    [3.0, 12.0, 3.0],
                    [2.5, 2.5, 3.3],
                ]
            ),
            0,
            torch.tensor(
                [
                    [-4.177099609326439, 7.969579237856498, 10.277535644199697],
                    [0.1856488604600999, -0.35420350057036537, -0.45677933473829735],
                ]
            ),
            torch.tensor(
                [[-71.8461, 137.0768, 176.7736], [-33.1290, 63.2076, 81.5123]]
            ),
        ],
    ],
)
def test_predict_pls(
    x: torch.Tensor,
    seed: int,
    particles: torch.Tensor,
    prediction: torch.Tensor,
):
    pls = PLS(
        basis=MockBasis(),
        cost=MockCost(),
    )
    set_seed(seed)
    predicted_samples = pls.predict_samples(
        particles=particles,
        x=x,
    ).detach()
    print(predicted_samples)
    assert torch.allclose(predicted_samples, prediction)
