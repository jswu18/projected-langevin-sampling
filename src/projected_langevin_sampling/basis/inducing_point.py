import math
from typing import Optional

import gpytorch
import torch

from src.kernels import PLSKernel
from src.projected_langevin_sampling.basis.base import PLSBasis
from src.samplers import sample_multivariate_normal


class InducingPointBasis(PLSBasis):
    """
    A Non-orthonormal basis (IP) approximation.
    The base class for projected Langevin sampling with particles on a function space approximated by a set of M inducing points.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        kernel: PLSKernel,
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        x_train: torch.Tensor,
    ):
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

    @property
    def approximation_dimension(self) -> int:
        """
        The dimensionality of the function space approximation M (number of inducing points).
        :return: The dimensionality of the function space approximation.
        """
        return self.x_induce.shape[0]

    def initialise_particles(
        self,
        number_of_particles: int,
        noise_only: bool = True,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        particle_noise = self._initialise_particles_noise(
            number_of_particles=number_of_particles,
            seed=seed,
        )
        return (
            particle_noise if noise_only else (self.y_induce[:, None] + particle_noise)
        )  # size (M, P)

    def calculate_untransformed_train_prediction_samples(
        self, particles: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the untransformed predictions of the particles on the training data used for cost calculations.
        :param particles: The particles of size (M, P).
        :return: The untransformed predictions of size (N, P).
        """
        return gpytorch.solve(
            lhs=self.base_gram_induce_train.to_dense().T,
            input=self.base_gram_induce,
            rhs=particles,
        )  #  k(X, Z) @ k(Z, Z)^{-1} @ U(t) of size (N, P)

    def calculate_energy_potential(
        self, particles: torch.Tensor, cost: torch.Tensor
    ) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, P).
        :param cost: The cost of size (P,).
        :return: The energy potential for each particle of size (P,).
        """
        inverse_base_gram_induce_particles = gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)

        # cost + M/2 * (k(Z, Z)^{-1} @ particle)^T (k(Z, Z)^{-1} @ particle)
        particle_energy_potential = (
            cost
            + self.approximation_dimension
            / 2
            * torch.square(inverse_base_gram_induce_particles).sum(dim=0)
        )  # size (P,)
        return particle_energy_potential.mean().item()

    def _calculate_particle_update(
        self,
        particles: torch.Tensor,
        cost_derivative: torch.Tensor,
        step_size: float,
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the Wasserstein projected Langevin sampling.
        :param particles: Particles of size (M, P).
        :param cost_derivative: The cost derivative of size (N, P).
        :param step_size: A step size for the projected Langevin sampling update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, P).
        """
        inverse_base_gram_particle_vector = gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)
        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(particles.shape[0]),
            cov=self.base_gram_induce,
            size=(particles.shape[1],),
        ).T  # e ~ N(0, k(Z, Z)) of size (M, P)

        # - eta * k(Z, X) @ d_2 c(Y, k(X, Z) @ k(Z, Z)^{-1} @ U(t))
        # - eta * M * k(Z, Z)^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        # size (M, P)
        particle_update = (
            -step_size * self.base_gram_induce_train @ cost_derivative
            - step_size
            * self.approximation_dimension
            * inverse_base_gram_particle_vector
            + math.sqrt(2.0 * step_size) * noise_vector
        ).detach()
        return particle_update.to_dense().detach()  # size (M, P)

    def sample_predictive_noise(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
    ):
        """
        Calculates the predictive noise for a given input.
        G([Z, x]) ~ N(0, r([Z, x], [Z, x]))
        :param particles: Particles of size (M, P)
        :param x: Test points of size (N*, D)
        :return: The predictive noise of size (N*, P)
        """
        # zx = torch.concatenate((self.x_induce, x), dim=0)  # (M+N*, D)
        # noise_covariance = self.kernel.forward(
        #     x1=zx,
        #     x2=zx,
        # )  # (M+N*, M+N*)
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
        return sample_multivariate_normal(
            mean=torch.zeros(noise_covariance.shape[0]),
            cov=noise_covariance,
            size=(particles.shape[1],),
        ).T  # (M+N*, P)

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
        )  # size (N*, P)