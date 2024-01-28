import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import PLSLinkFunction


class BernoulliCost(PLSCost):
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
    ) -> torch.distributions.Bernoulli:
        predictions = prediction_samples.mean(dim=1)
        return torch.distributions.Bernoulli(
            probs=predictions,
        )

    def calculate_cost(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )
        return (
            -self.y_train[:, None] * torch.log(train_prediction_samples)
            - (1 - self.y_train[:, None]) * torch.log(1 - train_prediction_samples)
        ).sum(dim=0)