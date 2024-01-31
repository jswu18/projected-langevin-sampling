from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.projected_langevin_sampling.link_functions import PLSLinkFunction


class PLSCost(ABC):
    """
    A base class for the cost function of the projected Langevin sampling.
    Methods pertaining to the cost function are implemented here such as calculating the cost,
    calculating the cost derivative, and transforming the prediction samples to the output space.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self, link_function: PLSLinkFunction, observation_noise: Optional[float] = None
    ):
        """
        Constructor for the base class of the cost function.
        :param link_function: The link function to transform the prediction samples to the output space.
        :param observation_noise: Optional observation noise, depends on the cost function.
        """
        self.observation_noise = observation_noise
        self.link_function = link_function

    @abstractmethod
    def predict(
        self,
        prediction_samples: torch.Tensor,
    ) -> torch.distributions.Distribution:
        """
        Constructs the predictive distribution from the prediction samples.
        :param prediction_samples: The prediction samples of size (N, J).
        :return: The predictive distribution.
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_cost(self, untransformed_train_prediction_samples) -> torch.Tensor:
        """
        Calculates the cost current particles. This method takes the untransformed train prediction
        samples calculated with the current particles. This is implemented in the basis class of PLS.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost of size (J,) for each particle.
        """
        raise NotImplementedError()

    @abstractmethod
    def calculate_cost_derivative(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the cost derivative of the untransformed train prediction samples. These are the prediction samples
        before being transformed by the link function.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost derivative of size (N, J).
        """
        raise NotImplementedError()

    def _calculate_cost_derivative_autograd(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Fallback autograd implementation of calculate_cost_derivative.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost derivative of size (N, J).
        """
        return (
            torch.vmap(
                torch.func.jacfwd(self.calculate_cost),
                in_dims=2,
            )(untransformed_train_prediction_samples[:, None, :])
            .reshape(untransformed_train_prediction_samples.T.shape)
            .T
        )

    def sample_observation_noise(
        self,
        number_of_particles: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Samples observation noise for a given number of particles.
        :param number_of_particles: The number of particles to sample noise for.
        :param seed: An optional seed for reproducibility.
        :return: A tensor of size (J, ).
        """
        if self.observation_noise is None:
            return torch.zeros(number_of_particles)
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        return torch.normal(
            mean=0.0,
            std=self.observation_noise,
            size=(number_of_particles,),
            generator=generator,
        ).flatten()

    def predict_samples(
        self,
        untransformed_samples: torch.Tensor,
        observation_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The link function applied to the untransformed samples plus observation noise. This is the prediction samples
        in the output space.
        :param untransformed_samples: The untransformed samples of size (N, J).
        :param observation_noise: Optional observation noise matrix of size (J,). If None, observation noise is sampled.
        :return: The prediction samples of size (N, J).
        """
        if observation_noise is None:
            observation_noise = self.sample_observation_noise(
                number_of_particles=untransformed_samples.shape[1]
            )
        return self.link_function(untransformed_samples + observation_noise[None, :])
