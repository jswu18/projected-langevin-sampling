import collections
from typing import List

import gpytorch
import numpy as np
import torch


def construct_average_gaussian_likelihood(
    likelihoods: List[gpytorch.likelihoods.GaussianLikelihood],
) -> gpytorch.likelihoods.GaussianLikelihood:
    """
    Construct an average Gaussian likelihood from a list of Gaussian likelihoods.
    :param likelihoods: A list of Gaussian likelihoods.
    :return: An average Gaussian likelihood.
    """
    average_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    average_likelihood.noise = torch.tensor(
        np.array(
            [likelihood.noise.cpu().detach().numpy() for likelihood in likelihoods]
        )
    ).mean()
    if torch.cuda.is_available():
        average_likelihood.cuda()
    return average_likelihood


def construct_average_ard_kernel(
    kernels: List[gpytorch.kernels.Kernel],
) -> gpytorch.kernels.Kernel:
    """
    Construct an ARD kernel with average parameters from a list of ARD kernels.
    :param kernels: A list of ARD kernels.
    :return: An average ARD kernel.
    """
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=kernels[0].base_kernel.ard_num_dims)
    )
    kernel.load_state_dict(
        collections.OrderedDict(
            {
                param: torch.tensor(
                    np.array(
                        [[k.state_dict()[param].mean(dim=0).cpu()] for k in kernels]
                    )
                    .mean(axis=0)
                    .reshape(kernel.state_dict()[param].shape)
                )
                for param in kernels[0].state_dict()
            }
        )
    )
    return kernel
