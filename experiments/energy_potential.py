from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EnergyPotentialBase(ABC):
    """
    A lightweight Numpy implementation of only the potential energy function used for compatibility with
    kernel goodness of fit tests.
    """

    def __init__(
        self,
        observation_noise: Optional[float],
        x_induce: np.ndarray,
        y_train: np.ndarray,
        base_gram_induce: np.ndarray,
        base_gram_induce_train: np.ndarray,
        jitter: float = 0.0,
    ):
        self.observation_noise: float = observation_noise
        self.jitter: float = jitter
        self.x_induce: np.ndarray = x_induce
        self.y_train: np.ndarray = y_train
        self.base_gram_induce = base_gram_induce  # k(Z, X) of size (M, M)
        self.base_gram_induce_train = base_gram_induce_train  # k(Z, X) of size (M, N)

    @property
    @abstractmethod
    def approximation_dimension(self) -> int:
        """
        The dimensionality of the function space approximation M.
        To be implemented by the child class. This is generally the number of inducing points.
        :return: The dimensionality of the function space approximation.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def transform(y: np.ndarray) -> np.ndarray:
        """
        The transformation of the data y depending on the output space.
        For regression this is the identity function. For classification this is the sigmoid function.
        :param y: The untransformed data.
        :return: The transformed data.
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_untransformed_train_predictions(
        self, particles: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the untransformed predictions of the particles on the training data used for cost calculations.
        :param particles: The particles of size (M, P).
        :return: The untransformed predictions of size (N, P).
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_cost(self, particles: np.ndarray) -> np.ndarray:
        """
        Calculates the cost for each particle.
        :return: An array of size (P, ).
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_energy_potential(self, particles: np.ndarray) -> np.ndarray:
        """
        Calculates the energy potential of the particles.
        :param particles: Particles of size (M, P).
        :return: The energy potential for each particle of size (P,).
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.calculate_energy_potential(*args, **kwargs)


class EnergyPotentialClassification(EnergyPotentialBase, ABC):
    def __init__(
        self,
        x_induce: np.ndarray,
        y_train: np.ndarray,
        base_gram_induce: np.ndarray,
        base_gram_induce_train: np.ndarray,
        jitter: float = 0.0,
    ):
        EnergyPotentialBase.__init__(
            self,
            observation_noise=None,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )

    @staticmethod
    def transform(y: np.ndarray) -> np.ndarray:
        return np.reciprocal(1 + np.exp(-y))

    def calculate_cost(self, particles: np.ndarray) -> np.ndarray:
        prediction = self.transform(
            self._calculate_untransformed_train_predictions(particles)
        )  # of size (N, P)
        return (
            -self.y_train[:, None] * np.log(prediction)
            - (1 - self.y_train[:, None]) * np.log(1 - prediction)
        ).sum(axis=0)


class EnergyPotentialRegression(EnergyPotentialBase, ABC):
    def __init__(
        self,
        observation_noise: float,
        x_induce: np.ndarray,
        y_train: np.ndarray,
        base_gram_induce: np.ndarray,
        base_gram_induce_train: np.ndarray,
        jitter: float = 0.0,
    ):
        EnergyPotentialBase.__init__(
            self,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )

    @staticmethod
    def transform(y: np.ndarray) -> np.ndarray:
        return y

    def calculate_cost(self, particles: np.ndarray) -> np.ndarray:
        prediction = self.transform(
            self._calculate_untransformed_train_predictions(particles=particles)
        )  # (N, P)

        # (1/sigma^2) * (k(X, Z) @ k(Z, Z)^{-1} @ U(t) - Y) of size (N, P)
        return (1 / (2 * self.observation_noise)) * np.square(
            prediction - self.y_train[:, None]
        ).sum(axis=0)


class EnergyPotentialOrthonormalBasis(EnergyPotentialBase, ABC):
    def __init__(
        self,
        observation_noise: Optional[float],
        x_induce: np.ndarray,
        y_train: np.ndarray,
        base_gram_induce: np.ndarray,
        base_gram_induce_train: np.ndarray,
        jitter: float = 0.0,
    ):
        EnergyPotentialBase.__init__(
            self,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(
            (1 / self.x_induce.shape[0]) * self.base_gram_induce
        )

        # Remove negative eigenvalues
        positive_eigenvalue_idx = np.where(self.eigenvalues > 0)[0]
        self.eigenvalues = self.eigenvalues[positive_eigenvalue_idx].real  # (K,)
        self.eigenvectors = self.eigenvectors[:, positive_eigenvalue_idx].real  # (M, K)
        # K is the number of eigenvalues to keep

        # Scale eigenvectors (M, K)
        self.scaled_eigenvectors = np.multiply(
            np.reciprocal(np.sqrt(self.approximation_dimension * self.eigenvalues))[
                None, :
            ],
            self.eigenvectors,
        )

    @property
    def approximation_dimension(self):
        return self.eigenvalues.shape[0]

    def _calculate_untransformed_train_predictions(
        self, particles: np.ndarray
    ) -> np.ndarray:
        return (
            self.base_gram_induce_train.T @ self.scaled_eigenvectors @ particles
        )  # k(X, Z) @ V_tilde @ U(t) of size (N, P)

    def calculate_energy_potential(self, particles: np.ndarray) -> np.ndarray:
        cost = self.calculate_cost(particles=particles)  # size (P, )

        particle_energy_potential = cost + 1 / 2 * np.multiply(
            particles,
            np.diag(np.reciprocal(self.eigenvalues)) @ particles,
        ).sum(
            axis=0
        )  # size (P,)
        return particle_energy_potential


class EnergyPotentialInducingPointBasis(EnergyPotentialBase, ABC):
    def __init__(
        self,
        observation_noise: Optional[float],
        x_induce: np.ndarray,
        y_train: np.ndarray,
        base_gram_induce: np.ndarray,
        base_gram_induce_train: np.ndarray,
        jitter: float = 0.0,
    ):
        EnergyPotentialBase.__init__(
            self,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )

    @property
    def approximation_dimension(self):
        return self.x_induce.shape[0]

    def _calculate_untransformed_train_predictions(
        self, particles: np.ndarray
    ) -> np.ndarray:
        return self.base_gram_induce_train.T @ np.linalg.solve(
            a=self.base_gram_induce,
            b=particles,
        )  #  k(X, Z) @ k(Z, Z)^{-1} @ U(t) of size (N, P)

    def calculate_energy_potential(self, particles: np.ndarray) -> np.ndarray:
        cost = self.calculate_cost(particles=particles)  # size (P, )

        inverse_base_gram_induce_particles = np.linalg.solve(
            self.base_gram_induce, particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)

        # cost + M/2 * (k(Z, Z)^{-1} @ particle)^T (k(Z, Z)^{-1} @ particle)
        particle_energy_potential = cost + self.approximation_dimension / 2 * np.square(
            inverse_base_gram_induce_particles
        ).sum(
            axis=0
        )  # size (P,)
        return particle_energy_potential


class EnergyPotentialClassificationIPB(
    EnergyPotentialInducingPointBasis, EnergyPotentialClassification
):
    def __init__(
        self,
        x_induce: np.ndarray,
        y_train: np.ndarray,
        base_gram_induce: np.ndarray,
        base_gram_induce_train: np.ndarray,
        jitter: float = 0.0,
    ):
        EnergyPotentialInducingPointBasis.__init__(
            self,
            observation_noise=None,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )
        EnergyPotentialClassification.__init__(
            self,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )


class EnergyPotentialClassificationONB(
    EnergyPotentialOrthonormalBasis, EnergyPotentialClassification
):
    def __init__(
        self,
        x_induce: np.ndarray,
        y_train: np.ndarray,
        base_gram_induce: np.ndarray,
        base_gram_induce_train: np.ndarray,
        jitter: float = 0.0,
    ):
        EnergyPotentialOrthonormalBasis.__init__(
            self,
            observation_noise=None,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )
        EnergyPotentialClassification.__init__(
            self,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )


class EnergyPotentialRegressionIPB(
    EnergyPotentialInducingPointBasis, EnergyPotentialRegression
):
    def __init__(
        self,
        observation_noise: float,
        x_induce: np.ndarray,
        y_train: np.ndarray,
        base_gram_induce: np.ndarray,
        base_gram_induce_train: np.ndarray,
        jitter: float = 0.0,
    ):
        EnergyPotentialInducingPointBasis.__init__(
            self,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )
        EnergyPotentialRegression.__init__(
            self,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )


class EnergyPotentialRegressionONB(
    EnergyPotentialOrthonormalBasis, EnergyPotentialRegression
):
    def __init__(
        self,
        observation_noise: float,
        x_induce: np.ndarray,
        y_train: np.ndarray,
        base_gram_induce: np.ndarray,
        base_gram_induce_train: np.ndarray,
        jitter: float = 0.0,
    ):
        EnergyPotentialOrthonormalBasis.__init__(
            self,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )
        EnergyPotentialRegression.__init__(
            self,
            observation_noise=observation_noise,
            x_induce=x_induce,
            y_train=y_train,
            base_gram_induce=base_gram_induce,
            base_gram_induce_train=base_gram_induce_train,
            jitter=jitter,
        )
