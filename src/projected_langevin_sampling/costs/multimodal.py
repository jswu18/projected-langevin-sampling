import gpytorch
import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import PLSLinkFunction


class MultiModalCost(PLSCost):
    """
    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        observation_noise: float,
        shift: float,
        bernoulli_noise: float,
        y_train: torch.Tensor,
        link_function: PLSLinkFunction,
    ):
        super().__init__(
            link_function=link_function, observation_noise=observation_noise
        )
        self.shift = shift
        self.bernoulli_noise = bernoulli_noise
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
        pass

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

        # (N, J)
        # y - (f(x) + c)
        # 0 - [-20, 20] - 10
        # -10 + [20, -20]
        # [10, -30]
        errors_mode_1 = self.y_train[:, None] - train_prediction_samples + self.shift
        # y - f(x)
        errors_mode_2 = self.y_train[:, None] - train_prediction_samples

        # (N, J)
        log_likelihood_mode_1 = -0.5 * (
            torch.square(errors_mode_1) / (self.observation_noise**2)
        ) - torch.log(
            torch.sqrt(2 * torch.tensor([torch.pi]) * (self.observation_noise**2))
        )

        log_likelihood_mode_2 = -0.5 * (
            torch.square(errors_mode_2) / (self.observation_noise**2)
        ) - torch.log(
            torch.sqrt(2 * torch.tensor([torch.pi]) * (self.observation_noise**2))
        )

        return -torch.logsumexp(
            torch.stack(
                [
                    torch.log(torch.tensor(self.bernoulli_noise))
                    + log_likelihood_mode_1,
                    torch.log(torch.tensor(1 - self.bernoulli_noise))
                    + log_likelihood_mode_2,
                ]
            ),
            dim=0,
        ).sum(axis=0)

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
        # if isinstance(self.link_function, IdentityLinkFunction):
        #     return self._calculate_cost_derivative_identity_link_function(
        #         untransformed_train_prediction_samples=untransformed_train_prediction_samples
        #     )
        return self._calculate_cost_derivative_autograd(
            untransformed_train_prediction_samples=untransformed_train_prediction_samples
        )
