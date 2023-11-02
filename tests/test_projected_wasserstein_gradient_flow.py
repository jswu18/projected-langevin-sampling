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
    )
    pwgf.particles = particles
    set_seed(seed)
    calculated_update = pwgf.update(
        learning_rate=torch.tensor(learning_rate),
        observation_noise=torch.tensor(observation_noise),
    ).detach()
    assert torch.allclose(calculated_update, update)


@pytest.mark.parametrize(
    "x,x_induce,y_induce,x_train,y_train,jitter,seed,number_of_samples,predict_noise",
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
            torch.tensor(
                [
                    [-4.177099609326439, 7.969579237856498, 10.277535644199697],
                    [0.1856488604600999, -0.35420350057036537, -0.45677933473829735],
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
    )
    set_seed(seed)
    sampled_predict_noise = pwgf.sample_predict_noise(
        x=x.double(),
        number_of_samples=number_of_samples,
    ).detach()
    assert torch.allclose(sampled_predict_noise, predict_noise)


@pytest.mark.parametrize(
    "x,x_induce,y_induce,x_train,y_train,jitter,seed,particles,prediction",
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
            torch.tensor(
                [
                    [48.17588315015214, -91.91581330880244, -118.53424158988344],
                    [15.674953704469171, -29.90658453326932, -38.56740418022185],
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
    )
    pwgf.particles = particles
    set_seed(seed)
    predicted_samples = pwgf.predict_samples(
        x=x.double(),
    ).detach()
    assert torch.allclose(predicted_samples, prediction)
