import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import (
    PLSLinkFunction,
    SquareLinkFunction,
)


class PoissonCost(PLSCost):
    """
    A class for the Poisson cost function of the projected Langevin sampling. This is a cost function for
    regression where the output space is R+.

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
        """
        Constructor for the Poisson cost function. This does not require any observation noise.
        :param y_train: The training labels of size (N,).
        :param link_function: The link function to transform the prediction samples to the output space R+.
        """
        super().__init__(link_function=link_function, observation_noise=None)
        self.y_train = y_train

    def predict(
        self,
        prediction_samples: torch.Tensor,
    ) -> torch.distributions.Poisson:
        """
        Constructs a Poisson distribution from the prediction samples.
        :param prediction_samples: The prediction samples of size (N, J).
        :return: The Poisson distribution where the rate is the mean of the prediction samples.
        """
        return torch.distributions.Poisson(
            rate=prediction_samples.mean(dim=1),
        )

    def calculate_cost(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the Poisson cost for the current particles. This method takes the untransformed train prediction
        samples calculated with the current particles. This is implemented in the basis class of PLS.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost of size (J,) for each particle.
        """
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

    def _calculate_cost_derivative_square_link_function(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        This method is used when the link function is the square.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost derivative of size (N, J).
        """
        return (
            -2
            * torch.divide(
                self.y_train[:, None], untransformed_train_prediction_samples
            )
            + 2 * untransformed_train_prediction_samples
        )

    def calculate_cost_derivative(
        self,
        untransformed_train_prediction_samples: torch.Tensor,
        force_autograd: bool = False,
    ) -> torch.Tensor:
        """
        Calculates the cost derivative of the untransformed train prediction samples. These are the prediction samples
        before being transformed by the link function. This method uses the autograd implementation if the link function
        is not the square link function.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :param force_autograd: An override to use autograd for the derivative calculation.
        :return: The cost derivative of size (N, J).
        """
        if isinstance(self.link_function, SquareLinkFunction) and not force_autograd:
            return self._calculate_cost_derivative_square_link_function(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
        else:
            return self._calculate_cost_derivative_autograd(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
