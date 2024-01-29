from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.projected_langevin_sampling.link_functions import PLSLinkFunction


class PLSCost(ABC):
    """

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self, link_function: PLSLinkFunction, observation_noise: Optional[float] = None
    ):
        self.observation_noise = observation_noise
        self.link_function = link_function

    @abstractmethod
    def predict(
        self,
        prediction_samples: torch.Tensor,
    ) -> torch.distributions.Distribution:
        raise NotImplementedError()

    @abstractmethod
    def calculate_cost(self, untransformed_train_prediction_samples) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def calculate_cost_derivative(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _calculate_cost_derivative_autograd(
        self, untransformed_train_prediction_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Fallback autograd implementation of calculate_cost_derivative.

        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, P).
        :return: The cost derivative of size (N, P).
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
        :return: A tensor of size (P, ).
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
        if observation_noise is None:
            observation_noise = self.sample_observation_noise(
                number_of_particles=untransformed_samples.shape[1]
            )
        return self.link_function(untransformed_samples + observation_noise[None, :])
