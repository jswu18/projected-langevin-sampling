import gpytorch
import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import PLSLinkFunction


class GaussianCost(PLSCost):
    """

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        observation_noise: float,
        y_train: torch.Tensor,
        link_function: PLSLinkFunction,
    ):
        super().__init__(
            link_function=link_function, observation_noise=observation_noise
        )
        self.y_train = y_train

    def predict(
        self,
        prediction_samples: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            mean=prediction_samples.mean(dim=1),
            covariance_matrix=torch.diag(prediction_samples.var(axis=1)),
        )

    def calculate_cost(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )
        # (1/sigma^2) * (k(X, Z) @ k(Z, Z)^{-1} @ U(t) - Y) of size (P)
        return (1 / (2 * self.observation_noise)) * torch.square(
            train_prediction_samples - self.y_train[:, None]
        ).sum(dim=0)
