from typing import Optional

import gpytorch
import torch

from src.gradient_flows.base import GradientFlowBase
from src.kernels import GradientFlowKernel


class GradientFlowRegression(GradientFlowBase):
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
        x_train: torch.Tensor,
        y_train: torch.Tensor,
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

    @staticmethod
    def transform(y: torch.Tensor) -> torch.Tensor:
        return y

    def _calculate_cost(self, particles: torch.Tensor):
        inverse_base_gram_particle_vector = gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)

        # (1/sigma^2) * (k(X, Z) @ k(Z, Z)^{-1} @ U(t) - Y) of size (N, P)
        return (1 / (2 * self.observation_noise)) * torch.square(
            self.base_gram_induce_train.T @ inverse_base_gram_particle_vector
            - self.y_train[:, None]
        ).sum(dim=0)

    def _calculate_cost_derivative(
        self,
        particles: torch.Tensor,
    ) -> torch.Tensor:
        """
        The derivative of the cost function with respect to the second component.
        :return: matrix of size (N, P)
        """
        inverse_base_gram_particle_vector = gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)

        # (1/sigma^2) * (k(X, Z) @ k(Z, Z)^{-1} @ U(t) - Y) of size (N, P)
        return (1 / self.observation_noise) * (
            self.base_gram_induce_train.T @ inverse_base_gram_particle_vector
            - self.y_train[:, None]
        )

    def predict(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        jitter: float = 1e-20,
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Predicts the mean and variance for a given input.
        :param x: input of size (N*, D)
        :param particles: particles of size (P, N)
        :param noise: noise of size (N, P)
        :param jitter: jitter to add to the diagonal of the covariance matrix if it is not positive definite
        :return: normal distribution of size (N*,)
        """
        samples = self.predict_samples(x=x, particles=particles, noise=noise)
        return gpytorch.distributions.MultivariateNormal(
            mean=samples.mean(dim=1),
            covariance_matrix=torch.diag(torch.clip(samples.var(axis=1), jitter, None)),
        )
