from typing import Tuple, Union

import gpytorch
import torch

from experiments.data import Data, ExperimentData
from src.gps import ExactGP, svGP
from src.kernels import PLSKernel
from src.projected_langevin_sampling import PLSRegressionONB
from src.projected_langevin_sampling.base.base import PLSBase


def load_pls(
    pls: PLSBase,
    model_path: str,
) -> (PLSRegressionONB, torch.Tensor):
    model_config = torch.load(model_path)
    particles = model_config["particles"]
    pls.observation_noise = model_config["observation_noise"]
    print(f"Loaded particles and observation_noise from {model_path=}.")
    return pls, particles


def load_svgp(
    model_path: str,
    x_induce: torch.Tensor,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    likelihood: Union[
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.likelihoods.BernoulliLikelihood,
    ],
    learn_inducing_locations: bool,
) -> Tuple[svGP, torch.Tensor]:
    model = svGP(
        x_induce=x_induce,
        mean=mean,
        kernel=kernel,
        learn_inducing_locations=learn_inducing_locations,
        likelihood=likelihood,
    )
    loaded_states = torch.load(model_path)
    model.load_state_dict(loaded_states["model"])
    print(f"Loaded svGP model from {model_path=}.")
    return model, loaded_states["losses"]


def load_ard_exact_gp_model(
    model_path: str,
    data_path: str,
    likelihood: gpytorch.likelihoods.Likelihood,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
) -> Tuple[ExactGP, torch.Tensor]:
    data = torch.load(data_path)
    data.x
    data.y
    model_state_dict = torch.load(model_path)
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
