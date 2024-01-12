from typing import Optional

import gpytorch
import torch

from src.gradient_flows.base import GradientFlowBase
from src.kernels import GradientFlowKernel


class GradientFlowBinaryClassification(GradientFlowBase):
    """
    N is the number of training points
    M is the number of inducing points
    P is the number of particles
    D is the dimensionality of the data
    """

    def __init__(
        self,
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        y_train: torch.Tensor,
        x_train: torch.Tensor,
        kernel: GradientFlowKernel,
        observation_noise: float,
        jitter: float = 0.0,
    ):
        super().__init__(
            x_induce=x_induce,
            y_train=y_train,
            x_train=x_train,
            kernel=kernel,
            observation_noise=observation_noise,
            y_induce=y_induce,
            jitter=jitter,
        )
        self.base_gram_induce = self.base_gram_induce[0, :, :]  # k(Z, X) of size (M, M)
        self.base_gram_induce_train = self.base_gram_induce_train[
            0, :, :
        ]  # k(Z, X) of size (M, N)

    @staticmethod
    def transform(y):
        return torch.div(1, 1 + torch.exp(-y))

    def _calculate_cost(self, particles: torch.Tensor):
        prediction = self.transform(
            gpytorch.solve(
                lhs=self.base_gram_induce_train.to_dense().T,
                input=self.base_gram_induce,
                rhs=particles,
            )
        )  # \phi(k(X, Z) @ k(Z, Z)^{-1} U(t)) of size (N, P)
        return (
            -self.y_train[:, None] * torch.log(prediction)
            - (1 - self.y_train[:, None]) * torch.log(1 - prediction)
        ).sum(dim=0)

    def _calculate_cost_derivative(
        self,
        particles: torch.Tensor,
    ) -> torch.Tensor:
        """
        The derivative of the cost function with respect to the second component.
        :return: matrix of size (N, P)
        """
        prediction = self.transform(
            gpytorch.solve(
                lhs=self.base_gram_induce_train.to_dense().T,
                input=self.base_gram_induce,
                rhs=particles,
            )
        )  # \phi(k(X, Z) @ k(Z, Z)^{-1} U(t)) of size (N, P)

        return -torch.mul(self.y_train[:, None], 1 - prediction) + torch.mul(
            1 - self.y_train[:, None], prediction
        )

    def predict(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Bernoulli:
        """
        :param x: input of size (N*, D)
        :param particles: particles of size (N, P)
        :param noise: noise of size (N*, P)
        :return: distribution of size (N*, 1)
        """
        samples = self.predict_samples(x=x, particles=particles, noise=noise)
        predictions = samples.mean(dim=1)
        # predictions[predictions > 0.5] = 1
        # predictions[predictions <= 0.5] = 0
        return torch.distributions.Bernoulli(
            probs=predictions,
        )
