from abc import ABC, abstractmethod
from typing import Optional

import torch


class PLSBasis(ABC):
    """
    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    @property
    def approximation_dimension(self) -> int:
        """
        The dimensionality of the function space approximation M.
        To be implemented by the child class. This is generally the number of inducing points.
        :return: The dimensionality of the function space approximation.
        """
        raise NotImplementedError

    def _initialise_particles_noise(
        self,
        number_of_particles: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Initialises the noise for each particle with a standard normal distribution.
        :param number_of_particles: The number of particles to initialise.
        :param seed: An optional seed for reproducibility.
        :return: A tensor of size (M, P).
        """
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=(
                self.approximation_dimension,
                number_of_particles,
            ),
            generator=generator,
        )  # size (M, P)

    @abstractmethod
    def initialise_particles(
        self,
        number_of_particles: int,
        noise_only: bool = True,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def calculate_untransformed_train_prediction_samples(
        self, particles: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the untransformed samples of the particles on the training data used for cost calculations.
        :param particles: The particles of size (M, P).
        :return: The untransformed predictions of size (N, P).
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_energy_potential(
        self, particles: torch.Tensor, cost: torch.Tensor
    ) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, P).
        :param cost: The cost of size (P,).
        :return: The energy potential for each particle of size (P,).
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_particle_update(
        self,
        particles: torch.Tensor,
        cost_derivative: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the Wasserstein projected Langevin sampling.
        :param particles: Particles of size (M, P).
        :param cost_derivative: The derivative of the cost function of size (N, P).
        :param step_size: A step size for the projected Langevin sampling update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, P).
        """
        raise NotImplementedError

    def calculate_particle_update(
        self,
        particles: torch.Tensor,
        cost_derivative: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the Wasserstein projected Langevin sampling.
        :param particles: Particles of size (M, P).
        :param cost_derivative: The derivative of the cost function of size (N, P).
        :param step_size: A step size for the projected Langevin sampling update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, P).
        """
        assert (
            particles.shape[0] == self.approximation_dimension
        ), f"Particles have shape {particles.shape} but requires ({self.approximation_dimension}, P) dimension."
        return self._calculate_particle_update(
            particles=particles,
            cost_derivative=cost_derivative,
            step_size=step_size,
        )

    @abstractmethod
    def sample_predictive_noise(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
    ):
        """
        Samples the predictive noise for a given input.
        :param particles: Particles of size (M, P)
        :param x: Test points of size (N*, D)
        :return: The predictive noise of size (N*, P)
        """
        raise NotImplementedError

    @abstractmethod
    def predict_untransformed_samples(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predicts samples for given test points x without applying the output transformation.
        :param particles: Particles of size (M, P).
        :param x: Test points of size (N*, D).
        :param noise: A noise tensor of size (N*, P), if None, it is sampled from the predictive noise distribution.
        :return: Predicted samples of size (N*, P).
        """
        raise NotImplementedError
