import math
from abc import ABC
from typing import Optional

import torch

from src.gradient_flows.base.base import GradientFlowBase
from src.kernels import GradientFlowKernel
from src.samplers import sample_multivariate_normal


class GradientFlowONBBase(GradientFlowBase, ABC):
    """
    The base class for gradient flows with particles on a function space approximated by an orthonormal basis (ONB)
    constructed from a set of M inducing points through the eigendecomposition of the kernel matrix.

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
        self.eigenvalues, self.eigenvectors = torch.linalg.eigh(
            self.base_gram_induce.evaluate()
        )

        # Remove negative eigenvalues
        positive_eigenvalue_idx = torch.where(self.eigenvalues > 0)[0]
        self.eigenvalues = self.eigenvalues[positive_eigenvalue_idx].real.double()
        self.eigenvectors = self.eigenvectors[:, positive_eigenvalue_idx].real.double()

        # Scale eigenvectors
        self.scaled_eigenvectors = torch.multiply(
            torch.divide(
                torch.ones(self.eigenvalues.shape),
                torch.sqrt(self.approximation_dimension * self.eigenvalues),
            )[None, :],
            self.eigenvectors,
        )

    @property
    def approximation_dimension(self):
        """
        The dimensionality of the function space approximation M (number of non-zero eigenvalues).
        :return: The dimensionality of the function space approximation.
        """
        return self.eigenvalues.shape[0]

    def initialise_particles(
        self,
        number_of_particles: int,
        noise_only: bool = True,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Initialise the particles with noise only.
        :param number_of_particles: The number of particles.
        :param noise_only: For ONB base, this is always True.
        :param seed: An optional seed for reproducibility.
        :return: The initialised particles.
        """
        if not noise_only:
            raise ValueError("For ONB base, noise_only must be True.")
        return self._initialise_particles_noise(
            number_of_particles=number_of_particles, seed=seed
        )

    def _calculate_untransformed_train_predictions(
        self, particles: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the untransformed predictions of the particles on the training data used for cost calculations.
        :param particles: The particles of size (M, P).
        :return: The untransformed predictions of size (N, P).
        """
        return (
            self.base_gram_induce_train.T @ self.scaled_eigenvectors @ particles
        )  # k(X, Z) @ V_tilde @ U(t) of size (N, P)

    def _calculate_untransformed_train_prediction(self, particles: torch.Tensor):
        """
        Calculates the untransformed prediction for each training point.
        :param particles: Particles of size (M, P).
        :return: The untransformed prediction of size (N, P).
        """
        return gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)

    def calculate_energy_potential(self, particles: torch.Tensor) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, P).
        :return: The energy potential for each particle of size (P,).
        """
        cost = self.calculate_cost(particles=particles)  # size (P, )

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

    def _calculate_particle_update(
        self,
        particles: torch.Tensor,
        learning_rate: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the update for each particle following the Wasserstein gradient flow.
        :param particles: Particles of size (M, P).
        :param learning_rate: A learning rate or step size for the gradient flow update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, P).
        """
        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(particles.shape[0]),
            cov=torch.eye(particles.shape[0]),
            size=(particles.shape[1],),
        ).T  # e ~ N(0, k(Z, Z)) of size (M, P)
        cost_derivative = self.calculate_cost_derivative(
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

        :param particles: Particles of size (M, P)
        :param x: Test points of size (N*, D)
        :return: The predictive noise of size (N*, P)
        """
        gram_x = self.kernel.forward(
            x1=x,
            x2=x,
        )
        base_gram_zx = self.kernel.base_kernel.forward(
            x1=self.x_induce,
            x2=x,
        )
        if base_gram_zx.ndim == 3:
            base_gram_zx = base_gram_zx[0, :, :]
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
        base_gram_x_induce = self.kernel.base_kernel.forward(
            x1=x,
            x2=self.x_induce,
        ).to_dense()
        if base_gram_x_induce.ndim == 3:
            base_gram_x_induce = base_gram_x_induce[0, :, :]

        # G([Z, x]) ~ N(0, R)
        if noise is None:
            noise = self.sample_predictive_noise(
                particles=particles,
                x=x,
            )

        return noise[self.approximation_dimension :, :] + (
            base_gram_x_induce
            @ self.scaled_eigenvectors
            @ (particles - noise[: self.approximation_dimension, :])
        )
