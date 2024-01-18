import torch

from src.kernels import PLSKernel
from src.projected_langevin_sampling.base.basis.inducing_point import (
    PLSInducingPointBasis,
)
from src.projected_langevin_sampling.base.transform.regression import PLSRegression


class PLSRegressionIPB(PLSInducingPointBasis, PLSRegression):
    """
    Projected Langevin sampling regression with particles on a function space approximated by a set of M inducing points.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        kernel: PLSKernel,
        observation_noise: float,
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        jitter: float = 0.0,
    ):
        PLSInducingPointBasis.__init__(
            self,
            kernel=kernel,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )
        PLSRegression.__init__(
            self,
            kernel=kernel,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )
