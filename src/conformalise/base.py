from abc import ABC, abstractmethod
from typing import Tuple, Union

import gpytorch.distributions
import numpy as np
import torch


class ConformaliseBase(ABC):
    def __init__(
        self,
        x_calibration: torch.Tensor,
        y_calibration: torch.Tensor,
    ):
        self.x_calibration = x_calibration
        self.y_calibration = y_calibration
        self.number_of_calibration_points = x_calibration.shape[0]

    @abstractmethod
    def _predict_uncalibrated_coverage(
        self,
        x: torch.Tensor,
        coverage: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns uncalibrated coverage predictions for a given input and coverage percentage
        Args:
            x: input data of shape (n, d)
            coverage: the coverage percentage

        Returns: Tuple of lower and upper bounds of shape (1, n)

        """
        raise NotImplementedError

    @abstractmethod
    def predict_median(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns median predictions for a given input.
        Args:
            x: input data of shape (n, d)

        Returns: median predictions of shape (1, n)

        """
        raise NotImplementedError

    def _calculate_calibration(self, coverage: float) -> float:
        """
        Calculates the calibration factor for a given coverage percentage.
        Args:
            coverage: the coverage percentage

        Returns: the calibration factor calculated with conformal prediction calibration

        """
        uncalibrated_lower, uncalibrated_upper = self._predict_uncalibrated_coverage(
            x=self.x_calibration,
            coverage=coverage,
        )
        scores = torch.max(
            torch.stack(
                [
                    uncalibrated_lower - self.y_calibration,
                    self.y_calibration - uncalibrated_upper,
                ],
                dim=1,
            ),
            dim=1,
        ).values
        calibration = torch.quantile(
            scores,
            float(
                np.clip(
                    (self.number_of_calibration_points + 1)
                    * coverage
                    / self.number_of_calibration_points,
                    0.0,
                    1.0,
                )
            ),
        )
        return calibration.double()

    def predict_coverage(
        self, x: torch.Tensor, coverage: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        calibration = self._calculate_calibration(coverage)
        uncalibrated_lower, uncalibrated_upper = self._predict_uncalibrated_coverage(
            x=x, coverage=coverage
        )
        calibrated_lower, calibrated_upper = (
            uncalibrated_lower - calibration,
            uncalibrated_upper + calibration,
        )
        median = self.predict_median(x)
        # nothing should cross the median
        return (
            torch.min(torch.stack([calibrated_lower, median], dim=1), dim=1).values,
            torch.max(torch.stack([calibrated_upper, median], dim=1), dim=1).values,
        )

    def calculate_average_interval_width(
        self,
        x: torch.Tensor,
        coverage: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        lower, upper = self.predict_coverage(x=x, coverage=coverage)
        return torch.mean(upper - lower)

    def predict_variance(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        lower, upper = self.predict_coverage(x=x, coverage=0.666)
        return (upper - lower) / 2

    def predict(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            mean=self.predict_median(x=x),
            covariance_matrix=torch.diag(self.predict_variance(x=x)),
        )

    def __call__(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return self.predict(x=x)
