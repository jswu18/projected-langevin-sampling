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
    model_path: str,
    base_kernel: gpytorch.kernels.Kernel,
    experiment_data: ExperimentData,
    induce_data: Data,
    jitter: float,
) -> (ProjectedWassersteinGradientFlow, torch.Tensor):
    model_config = torch.load(model_path)
    particles = model_config["particles"].to(torch.double)
    observation_noise = model_config["observation_noise"]
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
        observation_noise=observation_noise,
    )
    pwgf.particles = particles
    print(f"Loaded PWGF model from {model_path=}.")
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
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    )
    loaded_states = torch.load(model_path)
    model.load_state_dict(loaded_states["model"])
    model.double()
    print(f"Loaded svGP model from {model_path=}.")
    return model


def load_gp_models_and_induce_data(
    induce_data_path: str,
    subsample_gp_models_path: str,
) -> (ExactGP, Data):
    induce_data = torch.load(induce_data_path)
    induce_data.x.to(torch.double)
    induce_data.y.to(torch.double)
    model_state_dicts = torch.load(subsample_gp_models_path)
    models = [
        ExactGP(
            x=induce_data.x,
            y=induce_data.y,
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            mean=gpytorch.means.ConstantMean(),
            kernel=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=induce_data.x.shape[1])
            ),
        )
        for _ in range(len(model_state_dicts))
    ]
    for model, state_dict in zip(models, model_state_dicts):
        model.load_state_dict(state_dict)
    print(
        f"Loaded model from {subsample_gp_models_path=} and inducing points from {induce_data_path=}."
    )
    return models, induce_data
