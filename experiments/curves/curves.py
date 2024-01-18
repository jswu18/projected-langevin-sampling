from abc import ABC, abstractmethod

import torch

from src.projected_langevin_sampling.base.transform.classification import (
    PLSClassification,
)


class Curve(ABC):
    seed: int

    @staticmethod
    @abstractmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def calculate_curve(self, x: torch.Tensor) -> torch.Tensor:
        curve = self._calculate_curve(x)
        return (curve - curve.mean()) / curve.std()

    def regression(
        self, x: torch.Tensor, sigma_true: float, seed: int = None
    ) -> torch.Tensor:
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        return (
            self.calculate_curve(x)
            + sigma_true
            * torch.normal(mean=0.0, std=1.0, generator=generator, size=x.shape)
        ).reshape(-1)

    # @staticmethod
    # def classification(y_curve: torch.Tensor) -> torch.Tensor:
    #     probabilities = PLSClassification.transform(
    #         y=y_curve,
    #     )
    #     return torch.Tensor(probabilities > 0.5)

    @staticmethod
    def classification(y_curve: torch.Tensor, seed: int = None) -> torch.Tensor:
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        probabilities = PLSClassification.transform(
            y=y_curve,
        )
        return torch.bernoulli(probabilities, generator=generator).type(torch.bool)


class Curve1(Curve):
    __name__ = "$y=2 \sin(0.35 \pi (x-3)^2) + x^2$"
    seed: int = 1

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.sin(((x - 3) ** 2) * 0.35 * torch.pi) + x**2


class Curve2(Curve):
    __name__ = "$y=2\sin(2\pi x)$"
    seed: int = 2

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.sin(2 * x * torch.pi)


class Curve3(Curve):
    __name__ = "$y=1.2 \cos(2 \pi x)$ + x^2"
    seed: int = 3

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 1.2 * torch.cos(x * (2 * torch.pi)) + x**2


class Curve4(Curve):
    __name__ = "$y=2\sin(1.5\pi x) + 0.6 \cos(4.5 \pi x) + \sin(3.5 \pi x)$"
    seed: int = 4

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return (
            2 * torch.sin(x * (1.5 * torch.pi))
            + 0.6 * torch.cos(x * (4.5 * torch.pi))
            + torch.sin(x * (3.5 * torch.pi))
        )


class Curve5(Curve):
    __name__ = "$y=2 \sin(2\pi x) + x$"
    seed: int = 5

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.sin(2 * torch.pi * x) + x


class Curve6(Curve):
    __name__ = "$y=2 \sin(\pi x) + 0.5x^3$"
    seed: int = 6

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.sin(torch.pi * x) + 0.5 * (x**3)


class Curve7(Curve):
    __name__ = "$y=4\sin(\pi x) + 2\sin(3 \pi x) -2x$"
    seed: int = 7

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 4 * torch.sin(x * torch.pi) + 2 * torch.sin(x * (3 * torch.pi)) - 2 * x


class Curve8(Curve):
    __name__ = "$y=2\cos(\pi x) + \sin(3 \pi x) -x^2$"
    seed: int = 8

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.cos(x * torch.pi) + torch.sin(x * (3 * torch.pi)) - x**2


class Curve9(Curve):
    __name__ = "$y=\sin(0.5 \pi (x-2)^2)$"
    seed: int = 9

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(((x - 2) ** 2) * 0.5 * torch.pi)


class Curve10(Curve):
    __name__ = "$y=3\sqrt{4-x^2} + \sin(\pi x)$"
    seed: int = 10

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 3 * torch.sqrt(4 - x**2) + torch.sin(torch.pi * x)


CURVE_FUNCTIONS = [
    Curve1(),
    Curve2(),
    Curve3(),
    Curve4(),
    Curve5(),
    Curve6(),
    Curve7(),
    Curve8(),
    Curve9(),
    Curve10(),
]
