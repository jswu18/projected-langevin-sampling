from typing import Tuple, Union

import scipy
import torch

from src.conformalise.base import ConformaliseBase
from src.gps import ExactGP, svGP


class ConformaliseGP(ConformaliseBase):
    def __init__(
        self,
        gp: Union[ExactGP, svGP],
        x_calibration: torch.Tensor,
        y_calibration: torch.Tensor,
    ):
        self.gp = gp
        super().__init__(
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    def _predict_uncalibrated_coverage(
        self,
        x: torch.Tensor,
        coverage: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns uncalibrated coverage predictions for a given input and coverage percentage by
        taking the corresponding quantiles of the Gaussian Process predictive distribution.
        :param x: Input data of shape (N, D).
        :param coverage: The coverage percentage.
        :return: Tuple of lower and upper bounds of shape (1, N).
        """
        prediction = self.gp.likelihood(self.gp(x))
        confidence_interval_scale = scipy.special.ndtri((coverage + 1) / 2)
        lower_bound = prediction.mean - confidence_interval_scale * torch.sqrt(
            prediction.variance
        )
        upper_bound = prediction.mean + confidence_interval_scale * torch.sqrt(
            prediction.variance
        )
        return lower_bound, upper_bound

    def predict_median(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns median predictions for a given input by taking the mean of the Gaussian Process
        predictive distribution.
        :param x: Input data of shape (N, D).
        :return: Median predictions of shape (1, N).
        """
        return self.gp(x).mean
