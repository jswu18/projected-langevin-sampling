import math
from abc import ABC
from typing import Optional

import gpytorch
import torch

from src.gradient_flows.base.base import GradientFlowBase
from src.kernels import GradientFlowKernel
from src.samplers import sample_multivariate_normal


class GradientFlowNONBBase(GradientFlowBase, ABC):
    """
    A Non-orthonormal basis (NONB) approximation.
    The base class for gradient flows with particles on a function space approximated by a set of M inducing points.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        kernel: GradientFlowKernel,
        observation_noise: float,
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        jitter: float = 0.0,
    ):
        """
        Constructor for the data approximation base class of gradient flows.
        :param x_induce: The inducing points of size (M, D).
        :param y_induce: The inducing points of size (M,).
        :param x_train: The training points of size (N, D).
        :param y_train: The training points of size (N,).
        :param kernel: The gradient flow kernel.
        :param observation_noise: The observation noise.
        :param jitter: A jitter for numerical stability.
        """
        super().__init__(
            kernel=kernel,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_induce=y_induce,
            x_train=x_train,
            y_train=y_train,
            jitter=jitter,
        )

    @property
    def approximation_dimension(self):
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
        """
        Initialise the particles with either noise only or noise and an initialisation from the inducing points.
        :param number_of_particles: The number of particles.
        :param noise_only: Whether to initialise the particles to the noise only. Defaults to True.
        :param seed: An optional seed for reproducibility.
        :return: The initialised particles.
        """
        noise = self._initialise_particles_noise(
            number_of_particles=number_of_particles, seed=seed
        )
        return noise if noise_only else (self.y_induce[:, None] + noise)  # size (M, P)

    def _calculate_untransformed_train_predictions(
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

    def calculate_energy_potential(self, particles: torch.Tensor) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, P).
        :return: The energy potential for each particle of size (P,).
        """
        cost = self.calculate_cost(particles=particles)  # size (P, )

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
        learning_rate: float,
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the Wasserstein gradient flow.
        :param particles: Particles of size (M, P).
        :param learning_rate: A learning rate or step size for the gradient flow update in the form of a scalar.
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
        cost_derivative = self.calculate_cost_derivative(
            particles=particles
        )  # size (N, P)

        # - eta * k(Z, X) @ d_2 c(Y, k(X, Z) @ k(Z, Z)^{-1} @ U(t))
        # - eta * M * k(Z, Z)^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        # size (M, P)
        particle_update = (
            -learning_rate * self.base_gram_induce_train @ cost_derivative
            - learning_rate
            * self.approximation_dimension
            * inverse_base_gram_particle_vector
            + math.sqrt(2.0 * learning_rate) * noise_vector
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
        )
        gram_xz = self.kernel.forward(
            x1=x,
            x2=self.x_induce,
        )
        noise_covariance = torch.concatenate(
            [
                torch.concatenate(
                    [
                        self.gram_induce,
                        gram_xz.T,
                    ],
                    dim=1,
                ),
                torch.concatenate(
                    [
                        gram_xz,
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
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predicts samples for given test points x without applying the output transformation.
        :param particles: Particles of size (M, P).
        :param x: Test points of size (N*, D).
        :param noise: A noise tensor of size (N*, P), if None, it is sampled from the predictive noise distribution.
        :return: Predicted samples of size (N*, P).
        """
        gram_x_induce = self.kernel.forward(
            x1=x,
            x2=self.x_induce,
        ).to_dense()  # r(x, Z) of size (N*, M)

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
                input=self.gram_induce,
                rhs=(particles - noise[: self.approximation_dimension, :]),
            )
        )  # size (N*, P)
