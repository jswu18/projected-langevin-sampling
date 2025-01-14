import math
from typing import Optional

import torch

from src.projected_langevin_sampling.basis.base import PLSBasis
from src.projected_langevin_sampling.kernel import PLSKernel
from src.samplers import sample_multivariate_normal


class OrthonormalBasis(PLSBasis):
    """
    The base class for projected Langevin sampling with particles on a function space approximated by an orthonormal basis (ONB)
    constructed from a set of M inducing points through the eigendecomposition of the kernel matrix.

    N is the number of training points.
    M is the number of inducing points.
    M_k is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        kernel: PLSKernel,
        x_induce: torch.Tensor,
        x_train: torch.Tensor,
        eigenvalue_threshold: float = 0.0,
        additional_predictive_noise_distribution: torch.distributions.Distribution
        | None = None,
    ):
        super().__init__(
            additional_predictive_noise_distribution=additional_predictive_noise_distribution
        )
        self.kernel = kernel
        self.x_induce = x_induce  # size (M, D)
        self.base_gram_induce = self.kernel.base_kernel(
            x1=x_induce, x2=x_induce
        )  # k(Z, X) of size (M, M)
        self.base_gram_induce_train = self.kernel.base_kernel(
            x1=x_induce, x2=x_train
        )  # k(Z, X) of size (M, N)
        # move to gpu if available
        if torch.cuda.is_available():
            self.base_gram_induce = self.base_gram_induce.to(device="cuda")
            self.base_gram_induce_train = self.base_gram_induce_train.to(device="cuda")
        self.eigenvalues, self.eigenvectors = torch.linalg.eigh(
            (1 / self.x_induce.shape[0]) * self.base_gram_induce.evaluate()
        )

        # Remove eigenvalues and eigenvectors below the threshold. By default, this will
        # remove all negative eigenvalues and corresponding eigenvectors.
        desired_eigenvalue_idx = torch.where(self.eigenvalues > eigenvalue_threshold)[0]
        self.eigenvalues = self.eigenvalues[desired_eigenvalue_idx].real  # (M_k,)
        self.eigenvectors = self.eigenvectors[
            :, desired_eigenvalue_idx
        ].real  # (M, M_k)
        # M_k is the number of eigenvalues to keep
        print(
            f"Number of eigenvalues kept: {self.eigenvalues.shape[0]} out of {x_induce.shape[0]}"
        )

        # Scale eigenvectors (M, M_k)
        self.scaled_eigenvectors = torch.multiply(
            torch.reciprocal(
                torch.sqrt(self.approximation_dimension * self.eigenvalues)
            )[None, :],
            self.eigenvectors,
        )

    @property
    def approximation_dimension(self) -> int:
        """
        The dimensionality of the function space approximation M_k (number of non-zero eigenvalues).
        :return: The dimensionality of the function space approximation.
        """
        return self.eigenvalues.shape[0]

    def _initialise_particles(
        self,
        number_of_particles: int,
        noise_only: bool = True,
        seed: int | None = None,
    ) -> torch.Tensor:
        """
        Initialises the particles for the projected Langevin sampling with noise only.
        :param number_of_particles: The number of particles to initialise.
        :param noise_only: Whether to initialise the particles with noise only. For ONB base, this must be True.
        :param seed: An optional seed for reproducibility.
        :return: A tensor of size (M_k, J).
        """
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
        :param particles: The particles of size (M_k, J).
        :return: The untransformed predictions of size (N, J).
        """
        return torch.Tensor(
            self.base_gram_induce_train.T @ self.scaled_eigenvectors @ particles
        )  # k(X, Z) @ V_tilde @ U(t) of size (N, J)

    def calculate_energy_potential(
        self, particles: torch.Tensor, cost: torch.Tensor
    ) -> float:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, J).
        :param cost: The cost of size (J,).
        :return: The average energy potential of the particles.
        """

        particle_energy_potential = cost + 1 / 2 * torch.multiply(
            particles,
            torch.diag(torch.reciprocal(self.eigenvalues)) @ particles,
        ).sum(
            dim=0
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
        :param step_size: A step size for the projected Langevin sampling update in the form of a scalar.
        :return: The update to be applied to the particles of size (M, J).
        """
        # TODO: just sample IID from univariate normal
        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(particles.shape[0]),
            cov=torch.eye(particles.shape[0]),
            size=(particles.shape[1],),
        ).T  # e ~ N(0, k(Z, Z)) of size (M, J)

        # - eta *  V_tilde @ k(Z, X) @ d_2 c(Y, k(X, Z) @ V_tilde @ U(t))
        # - eta * Lambda^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        # size (M_k, J)
        particle_update = (
            -step_size
            * self.scaled_eigenvectors.T
            @ self.base_gram_induce_train
            @ cost_derivative
            - step_size * torch.diag(torch.reciprocal(self.eigenvalues)) @ particles
            + math.sqrt(2.0 * step_size) * noise_vector
        ).detach()
        return particle_update.to_dense().detach()  # size (M, J)

    def sample_predictive_noise(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the predictive noise for a given input.

        :param particles: Particles of size (M, J)
        :param x: Test points of size (N*, D)
        :return: The predictive noise of size (M+N*, J)
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
        base_gram_x_induce = self.kernel.base_kernel(
            x1=x,
            x2=self.x_induce,
        ).to_dense()
        if torch.cuda.is_available():
            base_gram_x_induce = base_gram_x_induce.cuda()
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
