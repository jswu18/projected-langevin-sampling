import torch

from src.distributions import NonParametric
from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import (
    LogisticGrowthLinkFunction,
    PLSLinkFunction,
)


class LogisticGrowthCost(PLSCost):
    """
    A class for the Logistic Growth cost function of the projected Langevin sampling. This is a cost function for
    regression where the output space is R.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        observation_noise: float,
        y_train: torch.Tensor,
        link_function: PLSLinkFunction,
    ):
        """
        Constructor for the Logistic Growth cost function.
        :param observation_noise: The observation noise.
        :param y_train: The training labels of size (N,).
        :param link_function: The link function to transform the prediction samples to the output space R.
        """
        super().__init__(link_function=link_function, observation_noise=None)
        self.y_train = y_train
        self.observation_noise = observation_noise

    def calculate_cost(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )

        # (J,)
        return (
            0.5
            * self.observation_noise
            * torch.square(
                self.y_train[:, None]
                - torch.multiply(train_prediction_samples, (1 - self.y_train[:, None]))
            ).sum(dim=0)
        )

    def _calculate_cost_derivative_logistic_growth_link_function(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        This method is used when the link function is the identity.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost derivative of size (N, J).
        """
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )
        return (
            0.5
            * self.observation_noise
            * torch.multiply(
                (
                    self.y_train[:, None]
                    - torch.multiply(
                        train_prediction_samples,
                        (1 - self.y_train[:, None]),
                    )
                ),
                (2 * train_prediction_samples - 1),
            )
        )

    def predict(self, prediction_samples: torch.Tensor) -> NonParametric:
        return NonParametric(samples=prediction_samples)

    def calculate_cost_derivative(
        self,
        untransformed_train_prediction_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the cost derivative of the untransformed train prediction samples. These are the prediction samples
        before being transformed by the link function. This method uses the autograd implementation if the link function
        is not the identity.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost derivative of size (N, J).
        """
        if isinstance(self.link_function, LogisticGrowthLinkFunction):
            return self._calculate_cost_derivative_logistic_growth_link_function(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
        else:
            return self._calculate_cost_derivative_autograd(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
