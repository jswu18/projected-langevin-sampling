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
        # return noise

    @abstractmethod
    def update(
        self,
        learning_rate: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

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

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        x: torch.Tensor,
        jitter: float = 1e-20,
    ) -> gpytorch.distributions.MultivariateNormal:
        raise NotImplementedError

    def __call__(self, x: torch.Tensor) -> gpytorch.distributions.Distribution:
        return self.predict(x=x)
