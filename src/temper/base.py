from abc import ABC, abstractmethod

import gpytorch
import torch


class TemperBase(ABC):
    def __init__(self, x_calibration: torch.Tensor, y_calibration: torch.Tensor):
        self.scale = self._calculate_scale(
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    def _calculate_scale(
        self, x_calibration: torch.Tensor, y_calibration: torch.Tensor
    ) -> float:
        y_prediction = self._untempered_predict(
            x=x_calibration,
        )

        # (2/N) * sum_i (y_i - m(x_i))^2 / sigma_i^2
        return 2 * torch.mean(
            torch.divide(
                torch.square(y_calibration - y_prediction.mean),
                torch.diag(y_prediction.covariance_matrix),
            )
        )

    @abstractmethod
    def _untempered_predict(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        prediction = self._untempered_predict(x=x)
        return gpytorch.distributions.MultivariateNormal(
            mean=prediction.mean,
            covariance_matrix=prediction.covariance_matrix * self.scale,
        )

    def __call__(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return self.predict(x=x)
