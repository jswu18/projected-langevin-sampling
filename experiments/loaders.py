import gpytorch
import torch

from experiments.data import Data, ExperimentData
from src.gps import ExactGP, svGP
from src.gradient_flows import ProjectedWassersteinGradientFlow
from src.kernels import GradientFlowKernel


def load_kernel_and_induce_data(
    induce_data_path: str,
    induce_gp_model_path: str,
    experiment_data: ExperimentData,
    gp_scheme: str,
) -> (gpytorch.models.GP, torch.Tensor):
    induce_data = torch.load(induce_data_path)
    induce_data.x.to(torch.double)
    induce_data.y.to(torch.double)

    if gp_scheme == "exact":
        model = ExactGP(
            x=induce_data.x,
            y=induce_data.y,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            mean=gpytorch.means.ConstantMean(),
            kernel=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=experiment_data.train.x.shape[1]
                )
            ),
        )
    elif gp_scheme == "svgp":
        model = svGP(
            x_induce=induce_data.x,
            mean=gpytorch.means.ConstantMean(),
            kernel=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=experiment_data.train.x.shape[1]
                )
            ),
            learn_inducing_locations=True,
        )
    else:
        raise ValueError(f"GP scheme {gp_scheme} not supported.")
    model.load_state_dict(torch.load(induce_gp_model_path))
    print(
        f"Loaded model from {induce_gp_model_path=} and inducing points from {induce_data_path=}."
    )
    return model, induce_data


def load_projected_wasserstein_gradient_flow(
    particle_path: str,
    base_kernel: gpytorch.kernels.Kernel,
    experiment_data: ExperimentData,
    induce_data: Data,
    jitter: float,
) -> (ProjectedWassersteinGradientFlow, torch.Tensor):
    particles = torch.load(particle_path).to(torch.double)
    pwgf = ProjectedWassersteinGradientFlow(
        number_of_particles=particles.shape[1],
        kernel=GradientFlowKernel(
            base_kernel=base_kernel,
            approximation_samples=experiment_data.train.x,
        ),
        x_induce=induce_data.x,
        y_induce=induce_data.y,
        x_train=experiment_data.train.x,
        y_train=experiment_data.train.y,
        jitter=jitter,
    )
    pwgf.particles = particles
    print(f"Loaded particles from {particle_path=}.")
    return pwgf


def load_svgp(
    model_path: str,
    x_induce: torch.Tensor,
    kernel: gpytorch.kernels.Kernel,
    learn_inducing_locations: bool,
) -> svGP:
    model = svGP(
        x_induce=x_induce.to(torch.double),
        mean=gpytorch.means.ConstantMean(),
        kernel=kernel,
        learn_inducing_locations=learn_inducing_locations,
    )
    model.load_state_dict(torch.load(model_path))
    model.double()
    print(f"Loaded svGP model from {model_path=}.")
    return model
