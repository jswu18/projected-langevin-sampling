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
        prediction = self.gp(x)
        confidence_interval_scale = scipy.special.ndtri((coverage + 1) / 2)
        lower_bound = prediction.mean - confidence_interval_scale * torch.sqrt(
            prediction.variance
        )
        upper_bound = prediction.mean + confidence_interval_scale * torch.sqrt(
            prediction.variance
        )
        return lower_bound, upper_bound

    def predict_median(self, x: torch.Tensor) -> torch.Tensor:
        return self.gp(x).mean
