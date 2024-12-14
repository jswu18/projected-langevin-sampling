import torch

from src.projected_langevin_sampling.basis.base import PLSBasis
from src.projected_langevin_sampling.costs.base import PLSCost


class PLS:
    """
    A class for the projected Langevin sampling model.
    This class contains the basis and cost defining the function space
    approximation and the desired output space respectively.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        basis: PLSBasis,
        cost: PLSCost,
        name: str | None = None,
    ):
        self.basis = basis
        self.cost = cost
        self.name: str = name if name is not None else "pls"

    @property
    def observation_noise(self) -> float:
        return self.cost.observation_noise

    @observation_noise.setter
    def observation_noise(self, value: float):
        self.cost.observation_noise = value

    def initialise_particles(
        self,
        number_of_particles: int,
        noise_only: bool = True,
        seed: int | None = None,
    ) -> torch.Tensor:
        return self.basis.initialise_particles(
            number_of_particles=number_of_particles,
            noise_only=noise_only,
            seed=seed,
        )

    def sample_observation_noise(
        self,
        number_of_particles: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Samples observation noise for a given number of particles.
        :param number_of_particles: The number of particles to sample noise for.
        :param seed: An optional seed for reproducibility.
        :return: A tensor of size (J, ).
        """
        return self.cost.sample_observation_noise(
            number_of_particles=number_of_particles,
            seed=seed,
        )

    def sample_predictive_noise(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
    ):
        return self.basis.sample_predictive_noise(
            particles=particles,
            x=x,
        )

    def calculate_cost(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cost of the particles.
        :param particles: The particles of size (M, J).
        :return: The cost of size (J,).
        """
        untransformed_train_prediction_samples = (
            self.basis.calculate_untransformed_train_prediction_samples(
                particles=particles,
            )
        )
        return self.cost.calculate_cost(
            untransformed_train_prediction_samples=untransformed_train_prediction_samples,
        )

    def calculate_cost_derivative(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cost derivative of the particles. Prediction samples are calculated with the current particles
        in the function space approximation. The cost derivative is then calculated with these function space prediction
        samples.
        :param particles: The particles of size (M, J).
        :return: The cost derivative of size (N, J).
        """
        untransformed_train_prediction_samples = (
            self.basis.calculate_untransformed_train_prediction_samples(
                particles=particles,
            )
        )
        return self.cost.calculate_cost_derivative(
            untransformed_train_prediction_samples=untransformed_train_prediction_samples,
        )

    def calculate_particle_update(
        self, particles: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the gradient flow. The update is calculated by first
        calculating the cost derivative and then applying the update rule defined in the basis of the function
        space approximation.
        :param particles:
        :param step_size:
        :return:
        """
        cost_derivative = self.calculate_cost_derivative(particles=particles)
        return self.basis.calculate_particle_update(
            particles=particles,
            cost_derivative=cost_derivative,
            step_size=step_size,
        )

    def calculate_energy_potential(self, particles: torch.Tensor) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, J).
        :return: The average energy potential of the particles.
        """
        assert (
            particles.shape[0] == self.basis.approximation_dimension
        ), f"Particles have shape {particles.shape} but requires ({self.basis.approximation_dimension}, J) dimension."
        cost = self.calculate_cost(particles=particles)
        return self.basis.calculate_energy_potential(
            particles=particles,
            cost=cost,
        )

    def predict_samples(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
        predictive_noise: torch.Tensor | None = None,
        observation_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predicts samples for given test points x and applies the output transformation.
        :param particles: Particles of size (M, J).
        :param x: Test points of size (N*, D).
        :param observation_noise: A noise tensor of size (N*, J), if None, it is sampled from the observation noise distribution.
        :param predictive_noise: A noise tensor of size (N*, J), if None, it is sampled from the predictive noise distribution.
        :return: Predicted samples of size (N*, J).
        """
        untransformed_samples = self.predict_untransformed_samples(
            particles=particles,
            x=x,
            noise=predictive_noise,
        )
        return self.cost.predict_samples(
            untransformed_samples=untransformed_samples,
            observation_noise=observation_noise,
        )

    def predict_untransformed_samples(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.basis.predict_untransformed_samples(
            particles=particles,
            x=x,
            noise=noise,
        )

    def predict(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        predictive_noise: torch.Tensor | None = None,
        observation_noise: torch.Tensor | None = None,
    ) -> torch.distributions.Distribution:
        prediction_samples = self.predict_samples(
            particles=particles,
            x=x,
            predictive_noise=predictive_noise,
            observation_noise=observation_noise,
        )
        return self.cost.predict(prediction_samples=prediction_samples)

    def __call__(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        predictive_noise: torch.Tensor | None = None,
        observation_noise: torch.Tensor | None = None,
    ) -> torch.distributions.Distribution:
        return self.predict(
            x=x,
            particles=particles,
            predictive_noise=predictive_noise,
            observation_noise=observation_noise,
        )
