import gpytorch
import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import PLSLinkFunction


class PoissonCost(PLSCost):
    """

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        y_train: torch.Tensor,
        link_function: PLSLinkFunction,
    ):
        super().__init__(link_function=link_function)
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
        return (
            -2
            * torch.multiply(
                self.y_train[:, None],
                torch.log(torch.abs(untransformed_train_prediction_samples)),
            )
            + train_prediction_samples
        ).sum(dim=0)
