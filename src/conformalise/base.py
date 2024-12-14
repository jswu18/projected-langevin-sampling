from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class ConformalPrediction:
    coverage: float
    mean: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


class ConformaliseBase(ABC):
    """
    Base class for conformal prediction methods that can be used to calibrate the
    predictive variance. Calibration uses a calibration set of (x, y) pairs
    where x is the input and y is the output.
    We follow the conformal prediction calibration method described in
    https://arxiv.org/abs/2107.07511
    """

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
        Returns uncalibrated coverage predictions for a given input and coverage percentage.
        :param x: Input data of shape (N, D).
        :param coverage: The coverage percentage.
        :return: Tuple of lower and upper bounds of shape (1, N).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_median(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns median predictions for a given input.
        :param x: Input data of shape (N, D).
        :return: Median predictions of shape (1, N).
        """
        raise NotImplementedError

    def _calculate_calibration(self, coverage: float) -> float:
        """
        Calculates the calibration factor for a given coverage percentage.
        :param coverage: The coverage percentage.
        :return: The calibration factor calculated with conformal prediction calibration.
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
        return calibration.item()

    def predict_coverage(
        self, x: torch.Tensor, coverage: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns coverage predictions for a given input and coverage percentage.
        :param x: Input data of shape (N, D).
        :param coverage: The coverage percentage.
        :return: Tuple of lower and upper bounds of shape (1, N).
        """
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
        coverage: float,
    ) -> float:
        """
        Calculates the average interval width for a given input and coverage percentage.
        :param x: Input data of shape (N, D).
        :param coverage: The coverage percentage.
        :return: The average interval width.
        """
        lower, upper = self.predict_coverage(x=x, coverage=coverage)
        return torch.mean(upper - lower).item()

    def predict_variance(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns variance predictions for a given input as
        half the width of the coverage interval at 2/3 coverage.
        :param x: Input data of shape (N, D).
        :return: Variance predictions of shape (1, N).
        """
        lower, upper = self.predict_coverage(x=x, coverage=2 / 3)
        return (upper - lower) / 2

    def predict(self, x: torch.Tensor, coverage: float) -> ConformalPrediction:
        """
        Returns a predictive distribution for a given input.
        :param x: Input data of shape (N, D).
        :param jitter: A small value to add to the diagonal of the covariance matrix.
        :return: Predictive distribution.
        """
        lower, upper = self.predict_coverage(x=x, coverage=coverage)
        return ConformalPrediction(
            coverage=coverage,
            mean=self.predict_median(x=x),
            lower=lower,
            upper=upper,
        )

    def __call__(self, x: torch.Tensor, coverage: float) -> ConformalPrediction:
        return self.predict(x=x, coverage=coverage)
