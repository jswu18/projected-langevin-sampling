from typing import Tuple

import gpytorch
import torch

from projected_langevin_sampling import PLS
from projected_langevin_sampling.gaussian_process import SVGP, ExactGP


def load_pls(
    pls: PLS,
    model_path: str,
) -> Tuple[PLS, torch.Tensor, float, int]:
    target_device = pls.basis.x_induce.device
    target_dtype = pls.basis.x_induce.dtype
    model_config = torch.load(
        model_path,
        map_location=target_device,
        weights_only=False,
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
    particles = particles.to(device=target_device, dtype=target_dtype)
    return pls, particles, best_lr, number_of_epochs


def load_svgp(
    model_path: str,
    x_induce: torch.Tensor,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    likelihood: gpytorch.likelihoods.Likelihood,
    learn_inducing_locations: bool,
) -> Tuple[SVGP, torch.Tensor, float]:
    model = SVGP(
        x_induce=x_induce,
        mean=mean,
        kernel=kernel,
        learn_inducing_locations=learn_inducing_locations,
        likelihood=likelihood,
    )
    loaded_states = torch.load(
        model_path,
        map_location=x_induce.device,
        weights_only=False,
    )
    model.load_state_dict(loaded_states["model"])
    model = model.to(device=x_induce.device, dtype=x_induce.dtype)
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
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tuple[ExactGP, torch.Tensor]:
    map_location = device if device is not None else "cpu"
    data = torch.load(data_path, map_location=map_location, weights_only=False)
    data.to(device=device, dtype=dtype)
    model_state_dict = torch.load(
        model_path,
        map_location=map_location,
        weights_only=False,
    )
    model = ExactGP(
        x=data.x,
        y=data.y,
        likelihood=likelihood.to(device=data.x.device, dtype=data.x.dtype),
        mean=mean.to(device=data.x.device, dtype=data.x.dtype),
        kernel=kernel.to(device=data.x.device, dtype=data.x.dtype),
    )
    model.load_state_dict(model_state_dict["model"])
    model = model.to(device=data.x.device, dtype=data.x.dtype)
    print(f"Loaded model from {model_path=} and from {data_path=}.")
    return model, model_state_dict["losses"]
