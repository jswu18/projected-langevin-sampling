import math
from typing import Optional

import gpytorch
import torch

from src.kernels import PLSKernel
from src.projected_langevin_sampling.basis.base import PLSBasis
from src.samplers import sample_multivariate_normal


class InducingPointBasis(PLSBasis):
    """
    A Non-orthonormal basis approximation of the function space using inducing points (IP).
    The base class for projected Langevin sampling with particles on a function space
    approximated by a set of M inducing points.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        kernel: PLSKernel,
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        x_train: torch.Tensor,
        additional_predictive_noise_distribution: Optional[
            torch.distributions.Distribution
        ] = None,
    ):
        super().__init__(
            additional_predictive_noise_distribution=additional_predictive_noise_distribution
        )
        self.kernel = kernel
        self.x_induce = x_induce  # size (M, D)
        self.y_induce = y_induce  # size (M,)
        self.gram_induce = self.kernel.forward(
            x1=x_induce, x2=x_induce
        )  # r(Z, Z) of size (M, M)
        self.base_gram_induce = self.kernel.base_kernel(
            x1=x_induce, x2=x_induce
        )  # k(Z, X) of size (M, M)
        self.base_gram_induce_train = self.kernel.base_kernel(
            x1=x_induce, x2=x_train
        )  # k(Z, X) of size (M, N)
        if torch.cuda.is_available():
            self.gram_induce = self.gram_induce.to(device="cuda")
            self.base_gram_induce = self.base_gram_induce.to(device="cuda")
            self.base_gram_induce_train = self.base_gram_induce_train.to(device="cuda")

    @property
    def approximation_dimension(self) -> int:
        """
        The dimensionality of the function space approximation M (number of inducing points).
        :return: The dimensionality of the function space approximation.
        """
        return self.x_induce.shape[0]

    def _initialise_particles(
        self,
        number_of_particles: int,
        noise_only: bool = True,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Initialises the particles for the projected Langevin sampling.
        :param number_of_particles: The number of particles to initialise.
        :param noise_only: Whether to initialise the particles with noise only or to add the inducing point values.
        :param seed: An optional seed for reproducibility.
        :return: A tensor of size (M, J).
        """
        particle_noise = self._initialise_particles_noise(
            number_of_particles=number_of_particles,
            seed=seed,
        )
        return (
            particle_noise if noise_only else (self.y_induce[:, None] + particle_noise)
        )  # size (M, J)

    def calculate_untransformed_train_prediction_samples(
        self, particles: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the untransformed predictions of the particles on the training data used for cost calculations.
        :param particles: The particles of size (M, J).
        :return: The untransformed predictions of size (N, J).
        """
        return gpytorch.solve(
            lhs=self.base_gram_induce_train.to_dense().T,
            input=self.base_gram_induce,
            rhs=particles,
        )  #  k(X, Z) @ k(Z, Z)^{-1} @ U(t) of size (N, J)

    def calculate_energy_potential(
        self, particles: torch.Tensor, cost: torch.Tensor
    ) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, J).
        :param cost: The cost of size (J,).
        :return: The average energy potential of the particles.
        """
        inverse_base_gram_induce_particles = gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, J)

        # cost + M/2 * (k(Z, Z)^{-1} @ particle)^T (k(Z, Z)^{-1} @ particle)
        particle_energy_potential = (
            cost
            + self.approximation_dimension
            / 2
            * torch.square(inverse_base_gram_induce_particles).sum(dim=0)
        )  # size (J,)
        return particle_energy_potential.mean().item()

    def _calculate_particle_update(
        self,
        particles: torch.Tensor,
        cost_derivative: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the Wasserstein projected Langevin sampling.
        :param particles: Particles of size (M, J).
        :param cost_derivative: The cost derivative of size (N, J).
        :param step_size: A step size for the projected Langevin sampling update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, J).
        """
        inverse_base_gram_particle_vector = gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, J)
        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(particles.shape[0]),
            cov=self.base_gram_induce,
            size=(particles.shape[1],),
        ).T  # e ~ N(0, k(Z, Z)) of size (M, J)

        # - eta * k(Z, X) @ d_2 c(Y, k(X, Z) @ k(Z, Z)^{-1} @ U(t))
        # - eta * M * k(Z, Z)^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        # size (M, J)
        particle_update = (
            -step_size * self.base_gram_induce_train @ cost_derivative
            - step_size
            * self.approximation_dimension
            * inverse_base_gram_particle_vector
            + math.sqrt(2.0 * step_size) * noise_vector
        ).detach()
        return particle_update.to_dense().detach()  # size (M, J)

    def sample_predictive_noise(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
    ):
        """
        Calculates the predictive noise for a given input.
        G([Z, x]) ~ N(0, r([Z, x], [Z, x]))
        :param particles: Particles of size (M, J)
        :param x: Test points of size (N*, D)
        :return: The predictive noise of size (N*, J)
        """
        gram_x = self.kernel.forward(
            x1=x,
            x2=x,
            additional_approximation_samples=x,
        )
        gram_induce_x = self.kernel.forward(
            x1=self.x_induce,
            x2=x,
            additional_approximation_samples=x,
        )
        noise_covariance = torch.concatenate(
            [
                torch.concatenate(
                    [
                        self.gram_induce,
                        gram_induce_x,
                    ],
                    dim=1,
                ),
                torch.concatenate(
                    [
                        gram_induce_x.T,
                        gram_x,
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )  # (M+N*, M+N*)
        predictive_noise = sample_multivariate_normal(
            mean=torch.zeros(noise_covariance.shape[0]),
            cov=noise_covariance,
            size=(particles.shape[1],),
        ).T  # (M+N*, J)
        if self.additional_predictive_noise_distribution is not None:
            predictive_noise += self.additional_predictive_noise_distribution.sample(
                predictive_noise.shape
            ).reshape(predictive_noise.shape)
        return predictive_noise

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

        # Use additional approximation samples to calculate the gram matrices to ensure better OOD predictive behaviour
        gram_x_induce = self.kernel.forward(
            x1=x, x2=self.x_induce, additional_approximation_samples=x
        ).to_dense()  # r(x, Z) of size (N*, M)
        gram_induce = self.kernel.forward(
            x1=self.x_induce, x2=self.x_induce, additional_approximation_samples=x
        ).to_dense()  # r(Z, Z) of size (M, M)

        # G([Z, x]) ~ N(0, r([Z, x], [Z, x]))
        if noise is None:
            noise = self.sample_predictive_noise(
                particles=particles,
                x=x,
            )

        # G(x) + r(x, Z) @ r(Z, Z)^{-1} @ (U(t)-G(Z))
        return noise[self.approximation_dimension :, :] + (
            gpytorch.solve(
                lhs=gram_x_induce,
                input=gram_induce,
                rhs=(particles - noise[: self.approximation_dimension, :]),
            )
        )  # size (N*, J)
