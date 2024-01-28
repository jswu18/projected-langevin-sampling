import math
from typing import Optional

import torch

from src.kernels import PLSKernel
from src.projected_langevin_sampling.basis.base import PLSBasis
from src.samplers import sample_multivariate_normal


class OrthonormalBasis(PLSBasis):
    """
    The base class for projected Langevin sampling with particles on a function space approximated by an orthonormal basis (ONB)
    constructed from a set of M inducing points through the eigendecomposition of the kernel matrix.

    N is the number of training points.
    M is the number of inducing points.
    K is the dimensionality of the function space approximation.
    P is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        kernel: PLSKernel,
        x_induce: torch.Tensor,
        x_train: torch.Tensor,
    ):
        self.kernel = kernel

        self.x_induce = x_induce  # size (M, D)
        self.base_gram_induce = self.kernel.base_kernel(
            x1=x_induce, x2=x_induce
        )  # k(Z, X) of size (M, M)
        self.base_gram_induce_train = self.kernel.base_kernel(
            x1=x_induce, x2=x_train
        )  # k(Z, X) of size (M, N)
        self.eigenvalues, self.eigenvectors = torch.linalg.eigh(
            (1 / self.x_induce.shape[0]) * self.base_gram_induce.evaluate()
        )

        # Remove negative eigenvalues
        positive_eigenvalue_idx = torch.where(self.eigenvalues > 0)[0]
        self.eigenvalues = self.eigenvalues[positive_eigenvalue_idx].real  # (K,)
        self.eigenvectors = self.eigenvectors[:, positive_eigenvalue_idx].real  # (M, K)
        # K is the number of eigenvalues to keep

        # Scale eigenvectors (M, K)
        self.scaled_eigenvectors = torch.multiply(
            torch.reciprocal(
                torch.sqrt(self.approximation_dimension * self.eigenvalues)
            )[None, :],
            self.eigenvectors,
        )

    @property
    def approximation_dimension(self) -> int:
        """
        The dimensionality of the function space approximation K (number of non-zero eigenvalues).
        :return: The dimensionality of the function space approximation.
        """
        return self.eigenvalues.shape[0]

    def initialise_particles(
        self,
        number_of_particles: int,
        noise_only: bool = True,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if not noise_only:
            raise ValueError("For ONB base, noise_only must be True.")
        return self._initialise_particles_noise(
            number_of_particles=number_of_particles,
            seed=seed,
        )

    def calculate_untransformed_train_prediction_samples(
        self, particles: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the untransformed predictions of the particles on the training data used for cost calculations.
        :param particles: The particles of size (K, P).
        :return: The untransformed predictions of size (N, P).
        """
        return (
            self.base_gram_induce_train.T @ self.scaled_eigenvectors @ particles
        )  # k(X, Z) @ V_tilde @ U(t) of size (N, P)

    def calculate_energy_potential(
        self, particles: torch.Tensor, cost: torch.Tensor
    ) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, P).
        :param cost: The cost of size (P,).
        :return: The average energy potential.
        """

        particle_energy_potential = cost + 1 / 2 * torch.multiply(
            particles,
            torch.diag(torch.reciprocal(self.eigenvalues)) @ particles,
        ).sum(
            dim=0
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
        :param step_size: A step size for the projected Langevin sampling update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, P).
        """
        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(particles.shape[0]),
            cov=torch.eye(particles.shape[0]),
            size=(particles.shape[1],),
        ).T  # e ~ N(0, k(Z, Z)) of size (M, P)

        # - eta *  V_tilde @ k(Z, X) @ d_2 c(Y, k(X, Z) @ V_tilde @ U(t))
        # - eta * Lambda^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        # size (K, P)
        particle_update = (
            -step_size
            * self.scaled_eigenvectors.T
            @ self.base_gram_induce_train
            @ cost_derivative
            - step_size * torch.diag(torch.reciprocal(self.eigenvalues)) @ particles
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

        :param particles: Particles of size (M, P)
        :param x: Test points of size (N*, D)
        :return: The predictive noise of size (N*, P)
        """
        # Use additional approximation samples to ensure better OOD predictive behaviour
        gram_x = self.kernel.forward(
            x1=x,
            x2=x,
            additional_approximation_samples=x,
        )
        base_gram_x_induce = self.kernel.base_kernel(
            x1=x,
            x2=self.x_induce,
        )
        off_diagonal_block = (
            base_gram_x_induce @ self.scaled_eigenvectors @ torch.diag(self.eigenvalues)
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
        base_gram_x_induce = self.kernel.base_kernel(
            x1=x,
            x2=self.x_induce,
        ).to_dense()
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