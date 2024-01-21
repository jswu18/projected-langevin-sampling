import torch

from src.kernels import PLSKernel
from src.projected_langevin_sampling.base.basis.inducing_point import (
    PLSInducingPointBasis,
)
from src.projected_langevin_sampling.base.transform.poisson_regression import (
    PLSPoissonRegression,
)


class PLSPoissonRegressionIPB(PLSInducingPointBasis, PLSPoissonRegression):
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
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        jitter: float = 0.0,
    ):
        PLSInducingPointBasis.__init__(
            self,
            kernel=kernel,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )
        PLSPoissonRegression.__init__(
            self,
            kernel=kernel,
            observation_noise=0.0,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )
