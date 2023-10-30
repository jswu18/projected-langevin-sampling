from typing import Optional

import gpytorch
import torch

from src.kernels import GradientFlowKernel
from src.samplers import sample_multivariate_normal


class ProjectedWassersteinGradientFlow:
    def __init__(
        self,
        x_induce: torch.Tensor,
        y_train: torch.Tensor,
        x_train: torch.Tensor,
        kernel: GradientFlowKernel,
        y_induce: Optional[torch.Tensor] = None,
        jitter: float = 0.0,
    ):
        self.kernel = kernel
        self.x_induce = x_induce
        self.y_induce = y_induce
        self.x_train = x_train
        self.y_train = y_train
        self.jitter = jitter

        self.gram_induce = self.kernel(x1=x_induce, x2=x_induce)  # r(Z, Z)
        self.base_gram_induce = self.kernel.base_kernel(
            x1=x_induce, x2=x_induce
        )  # k(Z, X)
        self.gram_induce_train = self.kernel(x1=x_induce, x2=x_train)  # r(Z, X)
        self.base_gram_induce_train = self.kernel.base_kernel(
            x1=x_induce, x2=x_train
        )  # k(Z, X)

    def initialise_particles(
        self,
        number_of_particles: int,
        seed: int = None,
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
                    self.x_induce.shape[0],
                    number_of_particles,
                ),
                generator=generator,
            )
        ).double()
        return (self.y_induce[:, None] + noise) if self.y_induce is not None else noise

    def calculate_update(
        self,
        particles: torch.Tensor,
        learning_rate: torch.Tensor,
        observation_noise: torch.Tensor,
    ) -> torch.Tensor:
        inverse_base_gram_particle_vector = gpytorch.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t)
        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(particles.shape[0]),
            cov=self.base_gram_induce,
            size=(particles.shape[1],),
        ).T  # e ~ N(0, k(Z, Z))

        # - (eta/sigma^2) * (k(Z, X) @ k(X, Z) @ k(Z, Z)^{-1} @ U(t) - k(Z, X) @ Y)
        # - eta * M * k(Z, Z)^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        return (
            -(learning_rate / observation_noise)
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
        )

    def _sample_predict_noise(
        self,
        gram_x: torch.Tensor,
        gram_x_induce: torch.Tensor,
        number_of_samples: int,
    ) -> torch.Tensor:
        # e(x) ~ N(0, r(x, x) - r(x, Z) @ r(Z, Z)^{-1} @ r(Z, x))
        return sample_multivariate_normal(
            mean=torch.zeros(gram_x.shape[0]),
            cov=(
                gram_x
                - gpytorch.solve(
                    lhs=gram_x_induce,
                    input=self.gram_induce,
                    rhs=gram_x_induce.T,
                )
            ),
            size=(number_of_samples,),
        ).T

    def sample_predict_noise(
        self,
        x: torch.Tensor,
        number_of_samples: int = 1,
    ) -> torch.Tensor:
        gram_x_induce = self.kernel(
            x1=x,
            x2=self.x_induce,
        )  # r(x, Z)
        gram_x = self.kernel(
            x1=x,
            x2=x,
        )  # r(x, x)
        return self._sample_predict_noise(
            gram_x=gram_x.evaluate(),
            gram_x_induce=gram_x_induce.evaluate(),
            number_of_samples=number_of_samples,
        )

    def predict(
        self,
        particles: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gram_x_induce = self.kernel(
            x1=x,
            x2=self.x_induce,
        ).evaluate()  # r(x, Z)
        gram_x = self.kernel(
            x1=x,
            x2=x,
        ).evaluate()  # r(x, x)
        noise_vector = self._sample_predict_noise(
            gram_x=gram_x,
            gram_x_induce=gram_x_induce,
            number_of_samples=particles.shape[1],
        )  # e(x)

        # r(x, Z) @ r(Z, Z)^{-1} @ U(t) + e(x)
        return (
            gpytorch.solve(
                lhs=gram_x_induce,
                input=self.gram_induce,
                rhs=particles,
            )
            + noise_vector
        )
