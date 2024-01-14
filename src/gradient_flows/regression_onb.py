from typing import Optional

import gpytorch
import torch

from src.gradient_flows.base import GradientFlowBase
from src.kernels import GradientFlowKernel
from src.samplers import sample_multivariate_normal


class GradientFlowRegressionONB(GradientFlowBase):
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
        super().__init__(
            x_induce=x_induce,
            y_train=y_train,
            x_train=x_train,
            kernel=kernel,
            observation_noise=observation_noise,
            y_induce=y_induce,
            jitter=jitter,
        )
        self.eigenvalues, self.eigenvectors = torch.linalg.eigh(
            self.base_gram_induce.evaluate()
        )

        # Remove negative eigenvalues
        positive_eigenvalue_idx = torch.where(self.eigenvalues > 0)[0]
        self.eigenvalues = self.eigenvalues[positive_eigenvalue_idx].real.double()
        self.eigenvectors = self.eigenvectors[:, positive_eigenvalue_idx].real.double()
        self.number_of_inducing_points = self.eigenvalues.shape[0]

        # Scale eigenvectors
        self.scaled_eigenvectors = torch.multiply(
            torch.divide(
                torch.ones(self.eigenvalues.shape),
                torch.sqrt(self.number_of_inducing_points * self.eigenvalues),
            )[None, :],
            self.eigenvectors,
        )

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
                    self.number_of_inducing_points,
                    number_of_particles,
                ),
                generator=generator,
            )
        ).double()  # size (M, P)
        return noise

    @staticmethod
    def transform(y: torch.Tensor) -> torch.Tensor:
        return y

    def _calculate_cost_derivative(
        self,
        particles: torch.Tensor,
    ) -> torch.Tensor:
        """
        The derivative of the cost function with respect to the second component.
        :return: matrix of size (N, P)
        """
        # (1/sigma^2) * (k(X, Z) @ V_tilde @ U(t) - Y) of size (N, P)
        return (1 / self.observation_noise) * (
            self.base_gram_induce_train.T @ self.scaled_eigenvectors @ particles
            - self.y_train[:, None]
        )

    def _calculate_cost(self, particles: torch.Tensor):
        # (1/sigma^2) * (k(X, Z) @ V_tilde @ U(t) - Y)^2 of size (N, P)
        return (1 / (2 * self.observation_noise)) * torch.square(
            self.base_gram_induce_train.T @ self.scaled_eigenvectors @ particles
            - self.y_train[:, None]
        ).sum(dim=0)

    def calculate_energy_potential(self, particles: torch.Tensor) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: particles of size (M, P)
        :return: loss of size (P,)
        """
        cost = self._calculate_cost(particles=particles)  # size (P, )

        particle_energy_potential = cost + 1 / 2 * torch.multiply(
            particles,
            torch.diag(
                torch.divide(torch.ones(self.eigenvalues.shape), self.eigenvalues)
            )
            @ particles,
        ).sum(
            dim=0
        )  # size (P,)
        return particle_energy_potential.mean().item()

    def calculate_particle_update(
        self,
        particles: torch.Tensor,
        learning_rate: torch.Tensor,
    ) -> torch.Tensor:
        assert (
            particles.shape[0] == self.number_of_inducing_points
        ), f"Particles have shape {particles.shape} but requires {self.number_of_inducing_points} inducing points."

        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(particles.shape[0]),
            cov=torch.eye(particles.shape[0]),
            size=(particles.shape[1],),
        ).T  # e ~ N(0, k(Z, Z)) of size (M, P)
        cost_derivative = self._calculate_cost_derivative(
            particles=particles
        )  # size (N, P)

        # - eta *  V_tilde @ k(Z, X) @ d_2 c(Y, k(X, Z) @ V_tilde @ U(t))
        # - eta * Lambda^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        # size (M, P)
        particle_update = (
            -learning_rate
            * self.scaled_eigenvectors.T
            @ self.base_gram_induce_train
            @ cost_derivative
            - learning_rate
            * torch.diag(
                torch.divide(torch.ones(self.eigenvalues.shape), self.eigenvalues)
            )
            @ particles
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

        :param particles: particles of size (M, P)
        :param x: input of size (N*, D)
        :return: predictive noise of size (N*, P)
        """
        gram_x = self.kernel.forward(
            x1=x,
            x2=x,
        )
        base_gram_zx = self.kernel.base_kernel.forward(
            x1=self.x_induce,
            x2=x,
        )
        off_diagonal_block = (
            base_gram_zx.T @ self.scaled_eigenvectors @ torch.diag(self.eigenvalues)
        )
        noise_covariance = torch.concatenate(
            [
                torch.concatenate(
                    [
                        torch.diag(self.eigenvalues),
                        off_diagonal_block.T,
                    ],
                    dim=1,
                ),
                torch.concatenate(
                    [
                        off_diagonal_block,
                        gram_x,
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )
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
        base_gram_x_induce = self.kernel.base_kernel.forward(
            x1=x,
            x2=self.x_induce,
        ).to_dense()

        # G([Z, x]) ~ N(0, R)
        if noise is None:
            noise = self.sample_predictive_noise(
                particles=particles,
                x=x,
            )

        return noise[self.number_of_inducing_points :, :] + (
            base_gram_x_induce
            @ self.scaled_eigenvectors
            @ (particles - noise[: self.number_of_inducing_points, :])
        )

    def predict(
        self,
        x: torch.Tensor,
        particles: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        jitter: float = 1e-20,
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Predicts the mean and variance for a given input.
        :param x: input of size (N*, D)
        :param particles: particles of size (P, N)
        :param noise: noise of size (N, P)
        :param jitter: jitter to add to the diagonal of the covariance matrix if it is not positive definite
        :return: normal distribution of size (N*,)
        """
        samples = self.predict_samples(x=x, particles=particles, noise=noise)
        return gpytorch.distributions.MultivariateNormal(
            mean=samples.mean(dim=1),
            covariance_matrix=torch.diag(torch.clip(samples.var(axis=1), jitter, None)),
        )
