from typing import Tuple

import torch

from src.conformalise.base import ConformaliseBase
from src.gradient_flows.regression import GradientFlowRegression


class ConformaliseGradientFlow(ConformaliseBase):
    def __init__(
        self,
        x_calibration: torch.Tensor,
        y_calibration: torch.Tensor,
        gradient_flow: GradientFlowRegression,
    ):
        self.gradient_flow = gradient_flow
        super().__init__(
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    def _predict_uncalibrated_coverage(
        self,
        x: torch.Tensor,
        coverage: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self.gradient_flow.predict_samples(
            x=x,
        )
        lower_quantile, upper_quantile = 0.5 - coverage / 2, 0.5 + coverage / 2
        lower_bound = torch.quantile(samples, q=lower_quantile, dim=1)
        upper_bound = torch.quantile(samples, q=upper_quantile, dim=1)
        return lower_bound, upper_bound

    def predict_median(self, x: torch.Tensor) -> torch.Tensor:
        samples = self.gradient_flow.predict_samples(
            x=x,
        )
        return torch.quantile(samples, q=0.5, dim=1)
