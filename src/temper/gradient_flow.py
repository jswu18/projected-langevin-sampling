import gpytorch
import torch

from src.gradient_flows.regression import GradientFlowRegression
from src.temper.base import TemperBase


class TemperGradientFlow(TemperBase):
    def __init__(
        self,
        gradient_flow: GradientFlowRegression,
        x_calibration: torch.Tensor,
        y_calibration: torch.Tensor,
    ):
        self.gradient_flow = gradient_flow
        super().__init__(
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    def _untempered_predict(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        return self.gradient_flow(x)
