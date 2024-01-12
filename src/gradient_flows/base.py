from abc import ABC, abstractmethod
from typing import Optional

import gpytorch
import torch

from src.kernels import GradientFlowKernel
from src.samplers import sample_multivariate_normal


class GradientFlowBase(ABC):
    """
    N is the number of training points
    M is the number of inducing points
    P is the number of particles
    D is the dimensionality of the data
    """

    def __init__(
        self,
        x_induce: torch.Tensor,
        y_induce: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        kernel: GradientFlowKernel,
        observation_noise: float,
        jitter: float = 0.0,
    ):
        self.observation_noise = observation_noise
        self.kernel = kernel
        self.x_induce = x_induce  # size (M, D)
        self.y_induce = y_induce  # size (M,)
        self.x_train = x_train  # size (N, D)
        self.y_train = y_train  # size (N,)
        self.jitter = jitter

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

    @staticmethod
    @abstractmethod
    def transform(y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def initialise_particles(
        self,
        number_of_particles: int,
        seed: Optional[int] = None,
        noise_only: Optional[bool] = False,
    ) -> torch.Tensor:
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        noise = (
            torch.normal(
                mean=0.0,
                std=1.0,
                size=(
                    self.y_induce.shape[0],
                    number_of_particles,
                ),
                generator=generator,
            )
        ).double()  # size (M, P)
        return noise if noise_only else (self.y_induce[:, None] + noise)  # size (M, P)

    @abstractmethod
    def _calculate_cost_derivative(self, particles) -> torch.Tensor:
        """
        The derivative of the cost function with respect to the second component.
        :return: matrix of size (N, P)
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_cost(self, particles) -> torch.Tensor:
        """
        The cost function for each particle.
        :return: matrix of size (P, )
        """
        raise NotImplementedError

    def calculate_energy_potential(self, particles: torch.Tensor) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: particles of size (M, P)
        :return: loss of size (P,)
        """
        cost = self._calculate_cost(particles=particles)  # size (P, )

        inverse_base_gram_induce_particles = gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)

        # cost + M/2 * (k(Z, Z)^{-1} @ particle)^T (k(Z, Z)^{-1} @ particle)
        particle_energy_potential = cost + self.x_induce.shape[0] / 2 * torch.square(
            inverse_base_gram_induce_particles
        ).sum(
            dim=0
        )  # size (P,)
        return particle_energy_potential.mean().item()

    def calculate_particle_update(
        self,
        particles: torch.Tensor,
        learning_rate: torch.Tensor,
    ) -> torch.Tensor:
        assert particles.shape[0] == self.x_induce.shape[0], (
            f"Particles have shape {particles.shape} but inducing points have shape "
            f"{self.x_induce.shape}"
        )

        inverse_base_gram_particle_vector = gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)
        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(particles.shape[0]),
            cov=self.base_gram_induce,
            size=(particles.shape[1],),
        ).T  # e ~ N(0, k(Z, Z)) of size (M, P)
        cost_derivative = self._calculate_cost_derivative(
            particles=particles
        )  # size (N, P)

        # - eta * k(Z, X) @ d_2 c(Y, k(X, Z) @ k(Z, Z)^{-1} @ U(t))
        # - eta * M * k(Z, Z)^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        # size (M, P)
        particle_update = (
            -learning_rate * self.base_gram_induce_train @ cost_derivative
            - learning_rate
            * (self.x_induce.shape[0])
            * inverse_base_gram_particle_vector
            + torch.sqrt(2.0 * learning_rate) * noise_vector
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

        :param particles: particles of size (M, P)
        :param x: input of size (N*, D)
        :return: predictive noise of size (N*, P)
        """
        zx = torch.concatenate((self.x_induce, x), dim=0)  # (M+N*, D)
        noise_covariance = self.kernel.forward(
            x1=zx,
            x2=zx,
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
        Predicts samples for a given input.

        :param particles: particles of size (M, P)
        :param x: input of size (N*, D)
        :param noise: noise of size (N*, P), if None, it is sampled from the predictive noise distribution
        :return: samples of size (N*, P)
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
        return noise[self.x_induce.shape[0] :, :] + (
            gpytorch.solve(
                lhs=gram_x_induce,
                input=self.gram_induce,
                rhs=(particles - noise[: self.x_induce.shape[0], :]),
            )
        )  # size (N*, P)

    def predict_samples(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predicts transformed samples for a given input.

        :param particles: particles of size (M, P)
        :param x: input of size (N*, D)
        :param noise: noise of size (N*, P), if None, it is sampled from the predictive noise distribution
        :return: samples of size (N*, P)
        """
        return self.transform(
            self.predict_untransformed_samples(
                particles=particles,
                x=x,
                noise=noise,
            )
        )

    @abstractmethod
    def predict(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Distribution:
        raise NotImplementedError

    def __call__(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Distribution:
        return self.predict(x=x, particles=particles, noise=noise)
