from typing import Union

import gpytorch
import torch

from src.gps import ExactGP, svGP
from src.temper.base import TemperBase


class TemperGP(TemperBase):
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

    def _untempered_predict(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        return self.gp(x)
