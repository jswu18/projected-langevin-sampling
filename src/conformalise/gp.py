from typing import Tuple, Union

import gpytorch
import scipy
import torch

from src.conformalise.base import ConformaliseBase
from src.gaussian_process.exact_gp import ExactGP
from src.gaussian_process.svgp import SVGP


class ConformaliseGP(ConformaliseBase):
    def __init__(
        self,
        gp: Union[ExactGP, SVGP],
        x_calibration: torch.Tensor,
        y_calibration: torch.Tensor,
    ):
        self.gp = gp
        super().__init__(
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    @property
    def likelihood(self):
        return self.gp.likelihood

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
        assert self.gp.likelihood is not None
        prediction = self.gp.likelihood(self.gp(x))
        confidence_interval_scale = scipy.stats.norm.interval(coverage)[1]
        lower_bound = prediction.mean - confidence_interval_scale * torch.sqrt(
            prediction.variance
        )
        upper_bound = prediction.mean + confidence_interval_scale * torch.sqrt(
            prediction.variance
        )
        if isinstance(self.gp.likelihood, gpytorch.likelihoods.GaussianLikelihood):
            return lower_bound, upper_bound
        elif isinstance(self.gp.likelihood, gpytorch.likelihoods.StudentTLikelihood):
            return lower_bound.mean(axis=0), upper_bound.mean(axis=0)
        else:
            raise ValueError(f"Unknown likelihood type: {type(self.gp.likelihood)=}")

    def predict_median(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns median predictions for a given input by taking the mean of the Gaussian Process
        predictive distribution.
        :param x: Input data of shape (N, D).
        :return: Median predictions of shape (1, N).
        """
        return torch.Tensor(self.gp(x).mean)
