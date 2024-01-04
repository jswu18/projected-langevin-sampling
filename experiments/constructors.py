from copy import deepcopy
from typing import List, Union

import gpytorch
import numpy as np
import torch

from experiments.data import Data
from src.conformalise import ConformaliseBase, ConformaliseGP, ConformaliseGradientFlow
from src.gps import svGP
from src.gradient_flows import GradientFlowRegression
from src.temper import TemperBase, TemperGP, TemperGradientFlow


def construct_tempered_model(
    model: Union[svGP, GradientFlowRegression],
    data: Data,
) -> TemperBase:
    if isinstance(model, svGP):
        return TemperGP(
            gp=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    elif isinstance(model, GradientFlowRegression):
        return TemperGradientFlow(
            gradient_flow=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    else:
        raise ValueError(f"Model type {type(model)} not supported")


def construct_conformalised_model(
    model: Union[svGP, GradientFlowRegression],
    data: Data,
) -> ConformaliseBase:
    if isinstance(model, svGP):
        return ConformaliseGP(
            gp=model,
            x_calibration=data.x,
            y_calibration=data.y,
        )
    elif isinstance(model, GradientFlowRegression):
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
        np.array([likelihood.noise.detach().numpy() for likelihood in likelihoods])
    ).mean()
    return average_likelihood


def construct_average_dirichlet_likelihood(
    likelihoods: List[gpytorch.likelihoods.DirichletClassificationLikelihood],
) -> gpytorch.likelihoods.DirichletClassificationLikelihood:
    average_likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
        targets=likelihoods[0].targets, learn_additional_noise=True
    )
    average_likelihood.noise = torch.tensor(
        np.array([likelihood.noise.detach().numpy() for likelihood in likelihoods])
    ).mean()
    return average_likelihood


def construct_average_ard_kernel(
    kernels: List[gpytorch.kernels.Kernel],
) -> gpytorch.kernels.Kernel:
    kernel = deepcopy(kernels[0])
    kernel.base_kernel.lengthscale = torch.concat(
        tensors=[k.base_kernel.lengthscale for k in kernels],
    ).mean(dim=0)
    kernel.outputscale = torch.tensor(
        np.array([k.outputscale.detach().numpy() for k in kernels])
    ).mean(dim=0)
    return kernel
