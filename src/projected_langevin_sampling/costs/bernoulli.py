import torch

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import (
    PLSLinkFunction,
    SigmoidLinkFunction,
)


class BernoulliCost(PLSCost):
    """
    A class for the Bernoulli cost function of the projected Langevin sampling. This is a cost function for
    binary classification where the output space is {0, 1}.

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
        Constructor for the Bernoulli cost function. This does not require any observation noise.
        :param y_train: The training labels of size (N,).
        :param link_function: The link function to transform the prediction samples to the output space {0, 1}.
        """
        super().__init__(link_function=link_function, observation_noise=None)
        self.y_train = y_train.type(torch.double)

    def predict(
        self,
        prediction_samples: torch.Tensor,
    ) -> torch.distributions.Bernoulli:
        """
        Constructs a Bernoulli distribution from the prediction samples.
        :param prediction_samples: The prediction samples of size (N, J).
        :return: The Bernoulli distribution.
        """
        predictions = prediction_samples.mean(dim=1)
        return torch.distributions.Bernoulli(
            probs=predictions,
        )

    def calculate_cost(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the cross entropy cost for the current particles. This method takes the untransformed train prediction
        samples calculated with the current particles. This is implemented in the basis class of PLS.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost of size (J,) for each particle.
        """
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )
        return -torch.log(train_prediction_samples).T @ self.y_train - torch.log(
            1 - train_prediction_samples
        ).T @ (1 - self.y_train)

    def _calculate_cost_derivative_sigmoid_link_function(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        This method is used when the link function is the sigmoid.
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
        is not the sigmoid.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost derivative of size (N, J).
        """
        if isinstance(self.link_function, SigmoidLinkFunction):
            return self._calculate_cost_derivative_sigmoid_link_function(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
        else:
            return self._calculate_cost_derivative_autograd(
                untransformed_train_prediction_samples=untransformed_train_prediction_samples
            )
