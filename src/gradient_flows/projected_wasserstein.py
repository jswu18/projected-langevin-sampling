from typing import Optional

import gpytorch
import torch

from src.kernels import GradientFlowKernel
from src.samplers import sample_multivariate_normal


class ProjectedWassersteinGradientFlow:
    """
    N is the number of training points
    M is the number of inducing points
    P is the number of particles
    D is the dimensionality of the data
    """

    def __init__(
        self,
        number_of_particles: int,
        x_induce: torch.Tensor,
        y_train: torch.Tensor,
        x_train: torch.Tensor,
        kernel: GradientFlowKernel,
        observation_noise: float,
        y_induce: Optional[torch.Tensor] = None,
        jitter: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.observation_noise = observation_noise
        self.number_of_particles = number_of_particles  # P
        self.particles = self.initialise_particles(
            number_of_particles=number_of_particles,
            x=x_induce,
            y=y_induce,
            seed=seed,
        )  # size (M, P)
        self.kernel = kernel
        self.x_induce = x_induce  # size (M, D)
        self.y_induce = y_induce  # size (M,)
        self.x_train = x_train  # size (N, D)
        self.y_train = y_train  # size (N,)
        self.jitter = jitter

        self.gram_induce = self.kernel(
            x1=x_induce, x2=x_induce
        )  # r(Z, Z) of size (M, M)
        self.base_gram_induce = self.kernel.base_kernel(
            x1=x_induce, x2=x_induce
        )  # k(Z, X) of size (M, M)
        self.gram_induce_train = self.kernel(
            x1=x_induce, x2=x_train
        )  # r(Z, X) of size (M, N)
        self.base_gram_induce_train = self.kernel.base_kernel(
            x1=x_induce, x2=x_train
        )  # k(Z, X) of size (M, N)

    def reset_particles(
        self,
        seed: Optional[int] = None,
    ):
        self.particles = self.initialise_particles(
            number_of_particles=self.number_of_particles,
            x=self.x_induce,
            y=self.y_induce,
            seed=seed,
        )  # size (M, P)

    @property
    def particles(self) -> torch.Tensor:
        return self._particles

    @particles.setter
    def particles(self, particles: torch.Tensor):
        self._particles = particles
        self.number_of_particles = particles.shape[1]

    @staticmethod
    def initialise_particles(
        number_of_particles: int,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
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
                    x.shape[0],
                    number_of_particles,
                ),
                generator=generator,
            )
        ).double()  # size (M, P)
        return (y[:, None] + noise) if y is not None else noise  # size (M, P)

    def update(
        self,
        learning_rate: torch.Tensor,
    ) -> torch.Tensor:
        inverse_base_gram_particle_vector = gpytorch.solve(
            self.base_gram_induce, self.particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)
        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(self.particles.shape[0]),
            cov=self.base_gram_induce,
            size=(self.number_of_particles,),
        ).T  # e ~ N(0, k(Z, Z)) of size (M, P)

        # - (eta/sigma^2) * (k(Z, X) @ k(X, Z) @ k(Z, Z)^{-1} @ U(t) - k(Z, X) @ Y)
        # - eta * M * k(Z, Z)^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        # size (M, P)
        particle_update = (
            -(learning_rate / self.observation_noise)
            * (
                self.base_gram_induce_train
                @ self.base_gram_induce_train.T
                @ inverse_base_gram_particle_vector
                - self.base_gram_induce_train @ self.y_train[:, None]
            )
            - learning_rate
            * (self.x_induce.shape[0])
            * inverse_base_gram_particle_vector
            + torch.sqrt(2.0 * learning_rate) * noise_vector
        ).detach()
        self.particles += particle_update
        return particle_update.to_dense().detach()  # size (M, P)

    def _sample_predict_noise(
        self,
        gram_x: torch.Tensor,
        gram_x_induce: torch.Tensor,
        number_of_samples: int,
        include_observation_noise: bool = True,
    ) -> torch.Tensor:
        """
        Samples from the predictive noise distribution.
        :param gram_x: gram matrix of size (N*, N*)
        :param gram_x_induce: gram matrix of size (N*, M)
        :param number_of_samples: number of samples to draw
        :return:
        """
        if include_observation_noise:
            cov = (
                gram_x
                - gpytorch.solve(
                    lhs=gram_x_induce,
                    input=self.gram_induce,
                    rhs=gram_x_induce.T,
                )
            ) + torch.tensor(self.observation_noise) * torch.eye(gram_x.shape[0])
        else:
            cov = gram_x - gpytorch.solve(
                lhs=gram_x_induce,
                input=self.gram_induce,
                rhs=gram_x_induce.T,
            )
        # e(x) ~ N(0, r(x, x) - r(x, Z) @ r(Z, Z)^{-1} @ r(Z, x) + sigma^2 I)
        return sample_multivariate_normal(
            mean=torch.zeros(gram_x.shape[0]),
            cov=cov,
            size=(number_of_samples,),
        ).T  # size (N*, number_of_samples)

    def sample_predict_noise(
        self,
        x: torch.Tensor,
        number_of_samples: int = 1,
        include_observation_noise: bool = True,
    ) -> torch.Tensor:
        """
        Samples from the predictive noise distribution for a given input.

        :param x: input of size (N*, D)
        :param number_of_samples: number of samples to draw
        :param include_observation_noise: whether to include observation noise
        :return: noise samples of size (N*, number_of_samples)
        """
        gram_x_induce = self.kernel(
            x1=x,
            x2=self.x_induce,
        )  # r(x, Z)
        gram_x = self.kernel(
            x1=x,
            x2=x,
        )  # r(x, x)
        return self._sample_predict_noise(
            gram_x=gram_x.to_dense(),
            gram_x_induce=gram_x_induce.to_dense(),
            number_of_samples=number_of_samples,
            include_observation_noise=include_observation_noise,
        )  # size (N*, number_of_samples)

    def predict_samples(
        self,
        x: torch.Tensor,
        include_observation_noise: bool = True,
    ) -> torch.Tensor:
        """
        Predicts samples for a given input.

        :param x: input of size (N*, D)
        :param include_observation_noise: whether to include observation noise in the sample
        :return: samples of size (N*, P)
        """
        gram_x_induce = self.kernel(
            x1=x,
            x2=self.x_induce,
        ).to_dense()  # r(x, Z) of size (N*, M)
        gram_x = self.kernel(
            x1=x,
            x2=x,
        ).to_dense()  # r(x, x) of size (N*, N*)
        noise_vector = self._sample_predict_noise(
            gram_x=gram_x,
            gram_x_induce=gram_x_induce,
            number_of_samples=self.number_of_particles,
            include_observation_noise=include_observation_noise,
        )  # e(x) of size (N*, P)

        # r(x, Z) @ r(Z, Z)^{-1} @ U(t) + e(x)
        # size (N*, P)
        return (
            gpytorch.solve(
                lhs=gram_x_induce,
                input=self.gram_induce,
                rhs=self.particles,
            )
            + noise_vector
        )

    def predict(
        self,
        x: torch.Tensor,
        jitter: float = 1e-20,
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Predicts the mean and variance for a given input.
        :param x: input of size (N*, D)
        :param jitter: jitter to add to the diagonal of the covariance matrix if it is not positive definite
        :return: normal distribution of size (N*,)
        """
        samples = self.predict_samples(x=x)
        return gpytorch.distributions.MultivariateNormal(
            mean=samples.mean(dim=1),
            covariance_matrix=torch.diag(torch.clip(samples.var(axis=1), jitter, None)),
        )

    def __call__(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return self.predict(x=x)
