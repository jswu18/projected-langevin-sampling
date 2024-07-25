import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import IdentityLinkFunction


class MockCost(PLSCost):
    """
    A mock cost used for testing projected Langevin sampling.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(self):
        super().__init__(
            link_function=IdentityLinkFunction(),
        )

    def predict(
        self, prediction_samples: torch.Tensor
    ) -> torch.distributions.Distribution:
        """
        Constructs the predictive distribution from the prediction samples.
        :param prediction_samples: The prediction samples of size (N, J).
        :return: The predictive distribution.
        """
        return torch.distributions.Normal(0, 1)

    def calculate_cost(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the cost current particles. This method takes the untransformed train prediction
        samples calculated with the current particles. This is implemented in the basis class of PLS.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost of size (J,) for each particle.
        """
        return torch.ones(untransformed_train_prediction_samples.shape[1])

    def calculate_cost_derivative(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the cost derivative of the untransformed train prediction samples. These are the prediction samples
        before being transformed by the link function.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost derivative of size (N, J).
        """
        return torch.ones_like(untransformed_train_prediction_samples)
