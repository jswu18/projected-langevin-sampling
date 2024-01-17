import collections
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
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=kernels[0].base_kernel.ard_num_dims)
    )
    kernel.load_state_dict(
        collections.OrderedDict(
            {
                param: torch.tensor(
                    np.array([[k.state_dict()[param].mean(dim=0)] for k in kernels])
                    .mean(axis=0)
                    .reshape(kernel.state_dict()[param].shape)
                )
                for param in kernels[0].state_dict()
            }
        )
    )
    return kernel
