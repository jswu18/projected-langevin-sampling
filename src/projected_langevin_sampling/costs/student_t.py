import gpytorch
import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import (
    IdentityLinkFunction,
    PLSLinkFunction,
)


class StudentTCost(PLSCost):
    """
    A class for the Student T cost function of the projected Langevin sampling. This is a cost function for
    regression where the output space is R.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        degrees_of_freedom: float,
        y_train: torch.Tensor,
        link_function: PLSLinkFunction,
    ):
        """
        Constructor for the Student T cost function. This requires the degrees of freedom.
        :param degrees_of_freedom: The degrees of freedom of the Student T distribution.
        :param y_train: The training labels of size (N,).
        :param link_function: The link function to transform the prediction samples to the output space R.
        """
        super().__init__(link_function=link_function, observation_noise=None)
        self.y_train = y_train
        self.degrees_of_freedom = degrees_of_freedom

    def predict(
        self,
        prediction_samples: torch.Tensor,
    ) -> torch.distributions.studentT.StudentT:
        """
        Constructs a Student T distribution from the prediction samples.
        :param prediction_samples: The prediction samples of size (N, J).
        :return: The Student T distribution.
        """
        return torch.distributions.studentT.StudentT(
            df=self.degrees_of_freedom,
            loc=float(prediction_samples.mean(dim=1)[0]),  # HACK THIS IS WRONG
            scale=float(
                torch.sqrt(prediction_samples.var(axis=1))[0]
            ),  # HACK THIS IS WRONG
        )

    def calculate_cost(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )

        # (J, N)
        errors = (train_prediction_samples - self.y_train[:, None]).T

        # (1/2)(degrees_of_freedom + 1) \log(1+\frac{(error)^2}{degrees_of_freedom}) + const of size (J,)
        return (
            0.5
            * (self.degrees_of_freedom + 1)
            * torch.log(1 + torch.square(errors) / self.degrees_of_freedom).sum(dim=1)
        )

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
        errors = train_prediction_samples - self.y_train[:, None]
        return (self.degrees_of_freedom + 1) * torch.divide(
            errors, (self.degrees_of_freedom + torch.square(errors))
        )

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
