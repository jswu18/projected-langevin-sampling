import pytest
import torch

from mockers.kernel import MockGradientFlowKernel, MockKernel
from src.gradient_flows import ProjectedWassersteinGradientFlow
from src.utils import set_seed


@pytest.mark.parametrize(
    "x_induce,y_induce,x_train,y_train,jitter,number_of_particles,seed,particles",
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
            torch.tensor([0.1, 2.3, 3.1, 2.1, 3.3]),
            0.0,
            3,
            0,
            torch.tensor(
                [
                    [3.640995979309082, 1.8065710067749023, -0.07878947257995605],
                    [3.8684312105178833, 2.2154775857925415, 1.9014045000076294],
                ]
            ).double(),
        ],
    ],
)
def test_initialise_particles(
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    jitter: float,
    number_of_particles: int,
    seed: int,
    particles: torch.Tensor,
):
    kernel = MockGradientFlowKernel(
        base_kernel=MockKernel(),
        approximation_samples=x_train,
    )
    pwgf = ProjectedWassersteinGradientFlow(
        number_of_particles=number_of_particles,
        seed=seed,
        kernel=kernel,
        x_induce=x_induce,
        y_induce=y_induce,
        x_train=x_train,
        y_train=y_train,
        jitter=jitter,
        observation_noise=1.0,
    )
    assert torch.allclose(pwgf.particles.detach(), particles)


@pytest.mark.parametrize(
    "x_induce,y_induce,x_train,y_train,jitter,seed,particles,learning_rate,observation_noise,update",
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
            torch.tensor([0.1, 2.3, 3.1, 2.1, 3.3]),
            0.0,
            0,
            torch.tensor(
                [
                    [3.640995979309082, 1.8065710067749023, -0.07878947257995605],
                    [3.8684312105178833, 2.2154775857925415, 1.9014045000076294],
                ]
            ).double(),
            0.1,
            1.0,
            torch.tensor(
                [
                    [-6.318556215166498, -3.3139141325395953, -26.852652634868846],
                    [-4.418808309172321, -4.0371477131348446, -43.87779529313816],
                ]
            ).double(),
        ],
    ],
)
def test_calculate_update(
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    jitter: float,
    seed: int,
    particles: torch.Tensor,
    learning_rate: float,
    observation_noise: float,
    update: torch.Tensor,
):
    kernel = MockGradientFlowKernel(
        base_kernel=MockKernel(),
        approximation_samples=x_train,
    )
    pwgf = ProjectedWassersteinGradientFlow(
        number_of_particles=particles.shape[1],
        kernel=kernel,
        x_induce=x_induce.double(),
        y_induce=y_induce.double(),
        x_train=x_train.double(),
        y_train=y_train.double(),
        jitter=jitter,
        observation_noise=observation_noise,
    )
    pwgf.particles = particles
    set_seed(seed)
    calculated_update = pwgf.update(
        learning_rate=torch.tensor(learning_rate),
    ).detach()
    assert torch.allclose(calculated_update, update)


@pytest.mark.parametrize(
    "x,x_induce,y_induce,x_train,y_train,jitter,seed,number_of_samples,observation_noise,predict_noise",
    [
        [
            torch.tensor(
                [
                    [3.0, 12.0, 3.0],
                    [2.5, 2.5, 3.3],
                ]
            ),
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
            torch.tensor([0.1, 2.3, 3.1, 2.1, 3.3]),
            0.0,
            0,
            3,
            1.0,
            torch.tensor(
                [
                    [-4.2839446519023285, 8.055917408067154, 10.468815164219743],
                    [-1.3521197663415734, -0.06432219118205296, 1.7156595598527962],
                ]
            ).double(),
        ],
    ],
)
def test_sample_predict_noise(
    x: torch.Tensor,
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    jitter: float,
    seed: int,
    number_of_samples: int,
    observation_noise: float,
    predict_noise: torch.Tensor,
):
    kernel = MockGradientFlowKernel(
        base_kernel=MockKernel(),
        approximation_samples=x_train,
    )
    pwgf = ProjectedWassersteinGradientFlow(
        number_of_particles=1,
        kernel=kernel,
        x_induce=x_induce.double(),
        y_induce=y_induce.double(),
        x_train=x_train.double(),
        y_train=y_train.double(),
        jitter=jitter,
        observation_noise=observation_noise,
    )
    set_seed(seed)
    sampled_predict_noise = pwgf.sample_predict_noise(
        x=x.double(),
        number_of_samples=number_of_samples,
    ).detach()
    assert torch.allclose(sampled_predict_noise, predict_noise)


@pytest.mark.parametrize(
    "x,x_induce,y_induce,x_train,y_train,jitter,seed,particles,observation_noise,prediction",
    [
        [
            torch.tensor(
                [
                    [3.0, 12.0, 3.0],
                    [2.5, 2.5, 3.3],
                ]
            ),
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
            torch.tensor([0.1, 2.3, 3.1, 2.1, 3.3]),
            0.0,
            0,
            torch.tensor(
                [
                    [-4.177099609326439, 7.969579237856498, 10.277535644199697],
                    [0.1856488604600999, -0.35420350057036537, -0.45677933473829735],
                ]
            ).double(),
            1.0,
            torch.tensor(
                [
                    [48.06903810757625, -91.82947513859177, -118.34296206986339],
                    [14.137185077667498, -29.61670322388101, -36.394965285630754],
                ]
            ).double(),
        ],
    ],
)
def test_predict(
    x: torch.Tensor,
    x_induce: torch.Tensor,
    y_induce: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    jitter: float,
    seed: int,
    particles: torch.Tensor,
    observation_noise: float,
    prediction: torch.Tensor,
):
    kernel = MockGradientFlowKernel(
        base_kernel=MockKernel(),
        approximation_samples=x_train,
    )
    pwgf = ProjectedWassersteinGradientFlow(
        number_of_particles=particles.shape[1],
        kernel=kernel,
        x_induce=x_induce.double(),
        y_induce=y_induce.double(),
        x_train=x_train.double(),
        y_train=y_train.double(),
        jitter=jitter,
        observation_noise=observation_noise,
    )
    pwgf.particles = particles
    set_seed(seed)
    predicted_samples = pwgf.predict_samples(
        x=x.double(),
    ).detach()
    assert torch.allclose(predicted_samples, prediction)
