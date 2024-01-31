import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import (
    PLSLinkFunction,
    SigmoidLinkFunction,
)


class BernoulliCost(PLSCost):
    """

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        y_train: torch.Tensor,
        link_function: PLSLinkFunction,
    ):
        super().__init__(link_function=link_function)
        self.y_train = y_train.type(torch.double)

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
        return -torch.log(train_prediction_samples).T @ self.y_train - torch.log(
            1 - train_prediction_samples
        ).T @ (1 - self.y_train)

    def _calculate_cost_derivative_sigmoid_link_function(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )
        return -torch.mul(
            self.y_train[:, None], 1 - train_prediction_samples
        ) + torch.mul(1 - self.y_train[:, None], train_prediction_samples)

    def calculate_cost_derivative(
        self,
        untransformed_train_prediction_samples: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.link_function, SigmoidLinkFunction):
            return self._calculate_cost_derivative_sigmoid_link_function(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
        else:
            return self._calculate_cost_derivative_autograd(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
