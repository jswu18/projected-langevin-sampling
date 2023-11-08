from typing import List, Union

import gpytorch
import torch

from experiments.data import Data
from src.conformalise import ConformaliseBase, ConformaliseGP, ConformaliseGradientFlow
from src.gps import svGP
from src.gradient_flows import ProjectedWassersteinGradientFlow
from src.temper import TemperBase, TemperGP, TemperGradientFlow


def construct_tempered_model(
    model: Union[svGP, ProjectedWassersteinGradientFlow],
    data: Data,
) -> TemperBase:
    if isinstance(model, svGP):
        return TemperGP(
            gp=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    elif isinstance(model, ProjectedWassersteinGradientFlow):
        return TemperGradientFlow(
            gradient_flow=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def construct_conformalised_model(
    model: Union[svGP, ProjectedWassersteinGradientFlow],
    data: Data,
) -> ConformaliseBase:
    if isinstance(model, svGP):
        return ConformaliseGP(
            gp=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    elif isinstance(model, ProjectedWassersteinGradientFlow):
        return ConformaliseGradientFlow(
            gradient_flow=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def construct_average_gaussian_likelihood(
    likelihoods: List[gpytorch.likelihoods.GaussianLikelihood],
) -> gpytorch.likelihoods.GaussianLikelihood:
    average_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    average_likelihood.noise = torch.tensor(
        [likelihood.noise for likelihood in likelihoods]
    ).mean()
    return average_likelihood


def construct_average_ard_kernel(
    kernels: List[gpytorch.kernels.Kernel],
) -> gpytorch.kernels.Kernel:
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(
            ard_num_dims=kernels[0].base_kernel.ard_num_dims,
        )
    )
    kernel.base_kernel.lengthscale = torch.concat(
        tensors=[k.base_kernel.lengthscale for k in kernels],
    ).mean(dim=0)
    kernel.outputscale = torch.tensor(
        data=[k.outputscale for k in kernels],
    ).mean(dim=0)
    return kernel
