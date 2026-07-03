from abc import ABC, abstractmethod

import gpytorch
import torch


class TemperBase(ABC):
    """
    Base class for tempering methods that can be used to calibrate the
    predictive variance. Calibration uses a calibration set of (x, y) pairs
    where x is the input and y is the output. The calibration set is used to
    calculate the scale factor that is used to temper the predictive variance.
    The scale factor is calculated as follows:

            scale = (2/N) * sum_i (y_i - m(x_i))^2 / sigma_i^2
    where N is the number of calibration points, m(x_i) is the mean of the
    predictive distribution at x_i, and sigma_i^2 is the variance of the
    predictive distribution at x_i.
    This is the closed form solution for the scale factor that minimizes the
    negative log likelihood of the calibration set.
    This is used for regression tasks.
    """

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
        return (
            2
            * torch.mean(
                torch.div(
                    torch.square(y_calibration - y_prediction.mean),
                    torch.diag(y_prediction.covariance_matrix),
                )
            ).item()
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
