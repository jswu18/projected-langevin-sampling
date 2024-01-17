from abc import ABC
from typing import Optional

import torch

from src.gradient_flows.base.base import GradientFlowBase
from src.kernels.gradient_flow_kernel import GradientFlowKernel


class GradientFlowClassificationBase(GradientFlowBase, ABC):
    """
    The base class for binary classification gradient flows.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        kernel: GradientFlowKernel,
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        jitter: float = 0.0,
    ):
        """
        Constructor for the data approximation base class of gradient flows.
        :param x_induce: The inducing points of size (M, D).
        :param y_induce: The inducing points of size (M,).
        :param x_train: The training points of size (N, D).
        :param y_train: The training points of size (N,).
        :param kernel: The gradient flow kernel.
        :param jitter: A jitter for numerical stability.
        """
        GradientFlowBase.__init__(
            self,
            kernel=kernel,
            observation_noise=None,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )

    @staticmethod
    def transform(y):
        return torch.div(1, 1 + torch.exp(-y))

    def predict(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        predictive_noise: Optional[torch.Tensor] = None,
        observation_noise: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Bernoulli:
        """
        Predicts a Bernoulli distribution for the given input.
        :param x: input of size (N*, D)
        :param particles: particles of size (P, N)
        :param predictive_noise: Optional predictive noise of size (N*, P)
        :param observation_noise: Optional observation noise of size (N*, P)
        :return: A Bernoulli distribution of size (N*, ).
        """
        samples = self.predict_samples(
            x=x,
            particles=particles,
            predictive_noise=predictive_noise,
            observation_noise=observation_noise,
        )
        predictions = samples.mean(dim=1)
        # predictions[predictions <= 0.5] = 0
        return torch.distributions.Bernoulli(
            probs=predictions,
        )

    def calculate_cost_derivative(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Calculates the derivative of the cost function with respect to the second component evaluated at each particle.
        :return: A tensor of size (N, P).
        """
        prediction = self.transform(
            self._calculate_untransformed_train_predictions(particles)
        )  # of size (N, P)

        return -torch.mul(self.y_train[:, None], 1 - prediction) + torch.mul(
            1 - self.y_train[:, None], prediction
        )

    def calculate_cost(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cost for each particle.
        :return: A tensor of size (P, ).
        """
        prediction = self.transform(
            self._calculate_untransformed_train_predictions(particles)
        )  # of size (N, P)
        return (
            -self.y_train[:, None] * torch.log(prediction)
            - (1 - self.y_train[:, None]) * torch.log(1 - prediction)
        ).sum(dim=0)
