import torch

from src.gradient_flows.base.basis.orthonormal_basis import GradientFlowONBBase
from src.gradient_flows.base.transforms.classification import (
    GradientFlowClassificationBase,
)
from src.kernels import GradientFlowONBKernel


class GradientFlowClassificationONB(
    GradientFlowONBBase, GradientFlowClassificationBase
):
    """
    Gradient Flow classification with particles on a function space approximated by an orthonormal basis.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        kernel: GradientFlowONBKernel,
        observation_noise: float,
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        jitter: float = 0.0,
    ):
        GradientFlowONBBase.__init__(
            self,
            kernel=kernel,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )
        GradientFlowClassificationBase.__init__(
            self,
            kernel=kernel,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )
