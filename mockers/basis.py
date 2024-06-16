import torch

from src.projected_langevin_sampling.basis.base import PLSBasis


class MockBasis(PLSBasis):
    """
    A mock basis used for testing projected Langevin sampling.

    N is the number of training points.
    M is the dimensionality of the function space approximation (fixed to 10 for this mock basis).
    J is the number of particles.
    D is the dimensionality of the data.
    """

    @property
    def approximation_dimension(self) -> int:
        return 10

    def _initialise_particles(
        self,
        number_of_particles: int,
        noise_only: bool = True,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Initialises the particles for the projected Langevin sampling.
        :param number_of_particles: The number of particles to initialise.
        :param noise_only: Whether to initialise the particles with noise only.
        :param seed: An optional seed for reproducibility.
        :return: A tensor of size (M, J).
        """
        return torch.ones((self.approximation_dimension, number_of_particles))

    def calculate_untransformed_train_prediction_samples(
        self, particles: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the untransformed samples of the particles on the training data used for cost calculations.
        :param particles: The particles of size (M, J).
        :return: The untransformed predictions of size (N, J).
        """
        return torch.ones((10, particles.shape[1]))

    def calculate_energy_potential(
        self, particles: torch.Tensor, cost: torch.Tensor
    ) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, J).
        :param cost: The cost of size (J,).
        :return: The average energy potential of the particles.
        """
        return 0.0

    def _calculate_particle_update(
        self,
        particles: torch.Tensor,
        cost_derivative: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the Projected Langevin sampling.
        :param particles: Particles of size (M, J).
        :param cost_derivative: The derivative of the cost function of size (N, J).
        :param step_size: A step size for the projected Langevin sampling update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, J).
        """
        return torch.ones_like(particles) + particles

    def sample_predictive_noise(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
    ):
        """
        Samples the predictive noise for a given input.
        :param particles: Particles of size (M, J)
        :param x: Test points of size (N*, D)
        :return: The predictive noise of size (N*, J)
        """
        return x @ torch.ones((x.shape[1], particles.shape[0])) @ particles

    def predict_untransformed_samples(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predicts samples for given test points x without applying the output transformation.
        :param particles: Particles of size (M, J).
        :param x: Test points of size (N*, D).
        :param noise: A noise tensor of size (N*, J), if None, it is sampled from the predictive noise distribution.
        :return: Predicted samples of size (N*, J).
        """
        return x @ torch.ones((x.shape[1], particles.shape[0])) @ particles
