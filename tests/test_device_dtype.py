import gpytorch
import pytest
import torch

from experiments.data import Data
from projected_langevin_sampling import PLS, PLSKernel
from projected_langevin_sampling.basis import OrthonormalBasis
from projected_langevin_sampling.costs import GaussianCost
from projected_langevin_sampling.inducing_point_selectors import (
    ConditionalVarianceInducingPointSelector,
)
from projected_langevin_sampling.link_functions import IdentityLinkFunction


def _construct_gaussian_pls(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> PLS:
    x = torch.linspace(0, 1, 6, device=device, dtype=dtype).reshape(-1, 1)
    y = torch.sin(x.reshape(-1))
    x_induce = x[::2]
    base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(
        device=device, dtype=dtype
    )
    base_kernel.base_kernel.lengthscale = torch.tensor(
        0.4,
        device=device,
        dtype=dtype,
    )
    base_kernel.outputscale = torch.tensor(1.2, device=device, dtype=dtype)
    pls_kernel = PLSKernel(
        base_kernel=base_kernel,
        approximation_samples=x_induce,
    )
    return PLS(
        basis=OrthonormalBasis(
            kernel=pls_kernel,
            x_induce=x_induce,
            x_train=x,
        ),
        cost=GaussianCost(
            observation_noise=0.2,
            y_train=y,
            link_function=IdentityLinkFunction(),
        ),
    )


def test_pls_initialise_particles_preserves_basis_dtype() -> None:
    pls = _construct_gaussian_pls(dtype=torch.float64)

    particles = pls.initialise_particles(number_of_particles=3, seed=0)

    assert particles.device.type == "cpu"
    assert particles.dtype == torch.float64


def test_observation_noise_follows_prediction_sample_dtype() -> None:
    cost = GaussianCost(
        observation_noise=0.2,
        y_train=torch.tensor([1.0, 2.0], dtype=torch.float64),
        link_function=IdentityLinkFunction(),
    )
    untransformed_samples = torch.ones((2, 3), dtype=torch.float64)

    prediction_samples = cost.predict_samples(
        untransformed_samples=untransformed_samples,
    )

    assert prediction_samples.dtype == untransformed_samples.dtype
    assert prediction_samples.device == untransformed_samples.device


def test_experiment_data_to_preserves_integer_label_dtype() -> None:
    data = Data(
        x=torch.ones((2, 1), dtype=torch.float32),
        y=torch.tensor([0, 1], dtype=torch.int64),
    )

    data.to(dtype=torch.float64)

    assert data.x.dtype == torch.float64
    assert data.y is not None
    assert data.y.dtype == torch.int64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pls_core_tensors_stay_on_cuda() -> None:
    pls = _construct_gaussian_pls(device="cuda", dtype=torch.float64)

    particles = pls.initialise_particles(number_of_particles=3, seed=0)
    update = pls.calculate_particle_update(particles=particles, step_size=1e-3)
    samples = pls.predict_samples(x=pls.basis.x_induce, particles=particles)

    assert particles.is_cuda
    assert update.is_cuda
    assert samples.is_cuda
    assert particles.dtype == torch.float64
    assert update.dtype == torch.float64
    assert samples.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_conditional_variance_selector_does_not_move_kernel_off_cuda() -> None:
    x = torch.linspace(0, 1, 6, device="cuda", dtype=torch.float64).reshape(-1, 1)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(
        device=x.device,
        dtype=x.dtype,
    )

    ConditionalVarianceInducingPointSelector()(x=x, m=3, kernel=kernel)

    assert next(kernel.parameters()).is_cuda
