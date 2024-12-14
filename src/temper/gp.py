from typing import Union

import gpytorch
import torch

from src.gaussian_process import ExactGP, svGP
from src.temper.base import TemperBase


class TemperGP(TemperBase):
    """
    Temper the predictive variance of a Gaussian Process.
    """

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
        """
        Predict the untempered predictive distribution. This is the predictive
        distribution of the Gaussian Process.
        :param x: Input data of shape (N, D).
        :return: The predictive distribution.
        """
        prediction = self.gp.likelihood(self.gp(x))
        assert isinstance(prediction, gpytorch.distributions.MultivariateNormal)
        return prediction
