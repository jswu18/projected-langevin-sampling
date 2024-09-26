from abc import ABC, abstractmethod

import torch


class PLSLinkFunction(ABC):
    """
    A base class for the link function of the projected Langevin sampling. This is used to transform the prediction
    samples to the output space.

    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    @abstractmethod
    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        Transforms the prediction samples to the output space.
        :param y: The prediction samples of size (N, J).
        :return: The transformed prediction samples of size (N, J).
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)


class ProbitLinkFunction(PLSLinkFunction):
    """
    The probit link function. This is the inverse CDF of the standard normal distribution.
    This transform can be used for binary classification where the output space is {0, 1}.
    """

    def __init__(self, jitter: float = 1e-10):
        self.jitter = jitter

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        # https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
        return torch.clip(
            (1 + torch.erf(y / torch.sqrt(torch.tensor(2.0)))) / 2,
            self.jitter,
            1 - self.jitter,
        )


class IdentityLinkFunction(PLSLinkFunction):
    """
    The identity link function.
    This transform can be used for regression where the output space is R.
    """

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return y


class SigmoidLinkFunction(PLSLinkFunction):
    """
    The sigmoid link function.
    This transform can be used for binary classification where the output space is {0, 1}.
    """

    def __init__(self, jitter: float = 1e-10):
        self.jitter = jitter

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return torch.clip(
            torch.reciprocal(1 + torch.exp(-y)), self.jitter, 1 - self.jitter
        )


class SquareLinkFunction(PLSLinkFunction):
    """
    The square link function.
    This transform can be used for regression where the output space is R+.
    """

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return torch.square(y)


class LogisticGrowthLinkFunction(PLSLinkFunction):
    """
    The logistic growth link function.
    This transform can be used for regression where the output space is R+.
    """

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return torch.multiply(y, 1 - y)
