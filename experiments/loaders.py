from typing import Tuple, Union

import gpytorch
import torch

from src.gaussian_process import ExactGP, svGP
from src.projected_langevin_sampling import PLS


def load_pls(
    pls: PLS,
    model_path: str,
) -> Tuple[PLS, torch.Tensor, float, int]:
    model_config = torch.load(
        model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    particles = model_config["particles"]
    pls.observation_noise = model_config["observation_noise"]
    print(f"Loaded particles and observation_noise from {model_path=}.")
    best_lr = None
    number_of_epochs = None
    if "best_lr" in model_config:
        best_lr = model_config["best_lr"]
    if "number_of_epochs" in model_config:
        number_of_epochs = model_config["number_of_epochs"]
    if torch.cuda.is_available():
        particles = particles.to(device="cuda")
    return pls, particles, best_lr, number_of_epochs


def load_svgp(
    model_path: str,
    x_induce: torch.Tensor,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    likelihood: Union[
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.likelihoods.BernoulliLikelihood,
        gpytorch.likelihoods.StudentTLikelihood,
    ],
    learn_inducing_locations: bool,
) -> Tuple[svGP, torch.Tensor, float]:
    model = svGP(
        x_induce=x_induce,
        mean=mean,
        kernel=kernel,
        learn_inducing_locations=learn_inducing_locations,
        likelihood=likelihood,
    )
    loaded_states = torch.load(
        model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.load_state_dict(loaded_states["model"])
    print(f"Loaded svGP model from {model_path=}.")
    best_learning_rate = None
    if "best_learning_rate" in loaded_states:
        best_learning_rate = loaded_states["best_learning_rate"]
    return model, loaded_states["losses"], best_learning_rate


def load_ard_exact_gp_model(
    model_path: str,
    data_path: str,
    likelihood: gpytorch.likelihoods.Likelihood,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
) -> Tuple[ExactGP, torch.Tensor]:
    data = torch.load(
        data_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    model_state_dict = torch.load(
        model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = ExactGP(
        x=data.x,
        y=data.y,
        likelihood=likelihood,
        mean=mean,
        kernel=kernel,
    )
    model.load_state_dict(model_state_dict["model"])
    print(f"Loaded model from {model_path=} and from {data_path=}.")
    return model, model_state_dict["losses"]
