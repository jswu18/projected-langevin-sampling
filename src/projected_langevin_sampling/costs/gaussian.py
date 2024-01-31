import gpytorch
import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import (
    IdentityLinkFunction,
    PLSLinkFunction,
)


class GaussianCost(PLSCost):
    """
    A class for the Gaussian cost function of the projected Langevin sampling. This is a cost function for
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
        Constructor for the Gaussian cost function. This requires observation noise.
        :param observation_noise: The observation noise.
        :param y_train: The training labels of size (N,).
        :param link_function: The link function to transform the prediction samples to the output space R.
        """
        super().__init__(
            link_function=link_function, observation_noise=observation_noise
        )
        self.y_train = y_train

    def predict(
        self,
        prediction_samples: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Constructs a multivariate normal distribution from the prediction samples.
        :param prediction_samples: The prediction samples of size (N, J).
        :return: The multivariate normal distribution.
        """
        return gpytorch.distributions.MultivariateNormal(
            mean=prediction_samples.mean(dim=1),
            covariance_matrix=torch.diag(prediction_samples.var(axis=1)),
        )

    def calculate_cost(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the negative log likelihood cost for the current particles. This method takes the untransformed train prediction
        samples calculated with the current particles. This is implemented in the basis class of PLS.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost of size (J,) for each particle.
        """
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )

        # (J, N)
        errors = (train_prediction_samples - self.y_train[:, None]).T

        # (1/sigma^2) * (k(X, Z) @ k(Z, Z)^{-1} @ U(t) - Y) of size (J)
        return (1 / (2 * self.observation_noise)) * torch.vmap(
            lambda x: x @ x,
        )(errors)

    def _calculate_cost_derivative_identity_link_function(
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
        return -torch.mul(
            self.y_train[:, None], 1 - train_prediction_samples
        ) + torch.mul(1 - self.y_train[:, None], train_prediction_samples)

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
        if isinstance(self.link_function, IdentityLinkFunction):
            return self._calculate_cost_derivative_identity_link_function(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
        else:
            return self._calculate_cost_derivative_autograd(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
