from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.kernels.gradient_flow_kernel import GradientFlowKernel
from src.samplers import sample_multivariate_normal


class GradientFlowBase(ABC):
    """
    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        kernel: GradientFlowKernel,
        observation_noise: Optional[float],
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        jitter: float = 0.0,
    ):
        """
        Constructor for the base class of gradient flows.
        :param kernel: The gradient flow kernel.
        :param observation_noise: The observation noise.
        :param x_induce: The inducing points of size (M, D).
        :param y_induce: The inducing points of size (M,).
        :param x_train: The training points of size (N, D).
        :param y_train: The training points of size (N,).
        :param jitter: A jitter for numerical stability.
        """
        self.observation_noise = observation_noise
        self.kernel = kernel
        self.jitter = jitter

        self.x_induce = x_induce  # size (M, D)
        self.y_induce = y_induce  # size (M,)
        self.x_train = x_train  # size (N, D)
        self.y_train = y_train  # size (N,)
        self.gram_induce = self.kernel.forward(
            x1=x_induce, x2=x_induce
        )  # r(Z, Z) of size (M, M)
        self.base_gram_induce = self.kernel.base_kernel(
            x1=x_induce, x2=x_induce
        )  # k(Z, X) of size (M, M)
        self.gram_induce_train = self.kernel.forward(
            x1=x_induce, x2=x_train
        )  # r(Z, X) of size (M, N)
        self.base_gram_induce_train = self.kernel.base_kernel(
            x1=x_induce, x2=x_train
        )  # k(Z, X) of size (M, N)

    @property
    @abstractmethod
    def approximation_dimension(self) -> int:
        """
        The dimensionality of the function space approximation M.
        To be implemented by the child class. This is generally the number of inducing points.
        :return: The dimensionality of the function space approximation.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def transform(y: torch.Tensor) -> torch.Tensor:
        """
        The transformation of the data y depending on the output space.
        For regression this is the identity function. For classification this is the sigmoid function.
        :param y: The untransformed data.
        :return: The transformed data.
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
        return (
            torch.normal(
                mean=0.0,
                std=1.0,
                size=(
                    self.approximation_dimension,
                    number_of_particles,
                ),
                generator=generator,
            )
        ).double()  # size (M, P)

    @abstractmethod
    def initialise_particles(
        self,
        number_of_particles: int,
        noise_only: bool = True,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Initialises the particles for the gradient flow.
        :param number_of_particles: The number of particles to initialise.
        :param noise_only: Whether to initialise the particles to the noise only. Defaults to True.
        :param seed: An optional seed for reproducibility.
        :return: a tensor of size (M, P).
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_untransformed_train_predictions(
        self, particles: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the untransformed predictions of the particles on the training data used for cost calculations.
        :param particles: The particles of size (M, P).
        :return: The untransformed predictions of size (N, P).
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_cost_derivative(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Calculates the derivative of the cost function with respect to the second component evaluated at each particle.
        :return: A tensor of size (N, P).
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_cost(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cost for each particle.
        :return: A tensor of size (P, ).
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_energy_potential(self, particles: torch.Tensor) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, P).
        :return: The energy potential for each particle of size (P,).
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_particle_update(
        self,
        particles: torch.Tensor,
        learning_rate: float,
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the Wasserstein gradient flow.
        To be implemented by the child class.
        :param particles: Particles of size (M, P).
        :param learning_rate: A learning rate or step size for the gradient flow update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, P).
        """
        raise NotImplementedError

    def calculate_particle_update(
        self,
        particles: torch.Tensor,
        learning_rate: float,
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the Wasserstein gradient flow.
        :param particles: Particles of size (M, P).
        :param learning_rate: A learning rate or step size for the gradient flow update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, P).
        """
        assert (
            particles.shape[0] == self.approximation_dimension
        ), f"Particles have shape {particles.shape} but requires ({self.approximation_dimension}, P) dimension."
        return self._calculate_particle_update(
            particles=particles,
            learning_rate=learning_rate,
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

    def sample_observation_noise(
        self,
        number_of_particles: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Samples observation noise for a given number of particles.
        :param number_of_particles: The number of particles to sample noise for.
        :param seed: An optional seed for reproducibility.
        :return: A tensor of size (N, P).
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

    def predict_samples(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
        observation_noise: torch.Tensor = None,
        predictive_noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predicts samples for given test points x and applies the output transformation.
        :param particles: Particles of size (M, P).
        :param x: Test points of size (N*, D).
        :param observation_noise: A noise tensor of size (N*, P), if None, it is sampled from the observation noise distribution.
        :param predictive_noise: A noise tensor of size (N*, P), if None, it is sampled from the predictive noise distribution.
        :return: Predicted samples of size (N*, P).
        """
        if observation_noise is None:
            observation_noise = self.sample_observation_noise(
                number_of_particles=particles.shape[1]
            )
        return self.transform(
            self.predict_untransformed_samples(
                particles=particles,
                x=x,
                noise=predictive_noise,
            )
            + observation_noise[None, :]
        )

    @abstractmethod
    def predict(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.distributions.Distribution:
        """
        Predicts the appropriate distribution for given test points x.
        :param x: Test points of size (N*, D).
        :param particles: Particles of size (M, P).
        :param noise: A noise tensor of size (N*, P), if None, it is sampled from the predictive noise distribution.
        :return: A distribution.
        """
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.distributions.Distribution:
        return self.predict(x=x, particles=particles, noise=noise)
