from abc import ABC, abstractmethod

import torch


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
        self, x: torch.Tensor, sigma_true: float, seed: int | None = None
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

    @staticmethod
    def classification(y_curve: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None
        probabilities = torch.reciprocal(1 + torch.exp(-y_curve))
        return torch.bernoulli(probabilities, generator=generator).type(torch.bool)


class Curve1(Curve):
    __name__ = "$y=2 \sin(0.35 \pi x^2)$"
    seed: int = 1

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.sin((x**2) * 0.35 * torch.pi)


class Curve2(Curve):
    __name__ = "$y=2\sin(1.5\pi x)$"
    seed: int = 2

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.sin(1.5 * x * torch.pi)


class Curve3(Curve):
    __name__ = "$y=1.2 \cos(1.5 \pi x)$ - 0.25x"
    seed: int = 3

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 1.2 * torch.cos(x * (1.5 * torch.pi)) - 0.25 * x


class Curve4(Curve):
    __name__ = "$y=2\sin(0.5\pi x) + 0.6 \cos(2 \pi x) + \sin\pi x)$"
    seed: int = 4

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return (
            2 * torch.sin(x * (0.5 * torch.pi))
            + 0.6 * torch.cos(x * (2 * torch.pi))
            + torch.sin(x * torch.pi)
        )


class Curve5(Curve):
    __name__ = "$y=2 \sin(1.5\pi x) + 0.25 x$"
    seed: int = 5

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.sin(1.5 * torch.pi * x) + 0.25 * x


class Curve6(Curve):
    __name__ = "$y=2 \sin(0.5\pi x^2) + 0.1x$"
    seed: int = 6

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 2 * torch.sin(0.5 * torch.pi * x**2) + 0.1 * x


class Curve7(Curve):
    __name__ = "$y=4\sin(\pi x) + 2\sin(2 \pi x) -x$"
    seed: int = 7

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 4 * torch.sin(x * torch.pi) + 2 * torch.sin(x * (2 * torch.pi)) - x


class Curve8(Curve):
    __name__ = "$y=6\cos(\pi x) + 3\sin(2 \pi x) -x^2$"
    seed: int = 8

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return 6 * torch.cos(x * torch.pi) + 3 * torch.sin(x * (2 * torch.pi)) - x**2


class Curve9(Curve):
    __name__ = "$y=\sin(0.3 \pi (x-2)^2) + 0.1x$"
    seed: int = 9

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(((x - 2) ** 2) * 0.3 * torch.pi) + 0.1 * x


class Curve10(Curve):
    __name__ = "$y=\sqrt{9-x^2} + \sin(\pi x)$"
    seed: int = 10

    @staticmethod
    def _calculate_curve(x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(9 - x**2) + torch.sin(torch.pi * x)


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
