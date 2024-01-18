import torch

from src.kernels import PLSKernel
from src.projected_langevin_sampling.base.basis.orthonormal import PLSOrthonormalBasis
from src.projected_langevin_sampling.base.transform.classification import (
    PLSClassification,
)


class PLSClassificationONB(PLSOrthonormalBasis, PLSClassification):
    """
    Projected Langevin sampling classification with particles on a function space approximated by an orthonormal basis.

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
        PLSOrthonormalBasis.__init__(
            self,
            kernel=kernel,
            observation_noise=None,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )
        PLSClassification.__init__(
            self,
            kernel=kernel,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )
