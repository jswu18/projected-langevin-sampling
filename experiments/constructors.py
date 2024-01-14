from copy import deepcopy
from typing import List

import gpytorch
import numpy as np
import torch


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


# def construct_average_ard_kernel(
#     kernels: List[gpytorch.kernels.Kernel],
# ) -> gpytorch.kernels.Kernel:
#     kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#     kernel.base_kernel.lengthscale = torch.concat(
#         tensors=[k.base_kernel.lengthscale for k in kernels],
#     ).mean()
#     kernel.outputscale = np.array(
#         [np.mean([k.outputscale.detach().numpy() for k in kernels])]
#     )
#     return kernel
