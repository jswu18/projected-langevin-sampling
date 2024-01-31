from typing import Tuple

import torch

from src.conformalise.base import ConformaliseBase
from src.projected_langevin_sampling import ProjectedLangevinSampling


class ConformaliseGradientFlow(ConformaliseBase):
    def __init__(
        self,
        x_calibration: torch.Tensor,
        y_calibration: torch.Tensor,
        pls: ProjectedLangevinSampling,
        particles: torch.Tensor,
    ):
        self.pls = pls
        self.particles = particles
        super().__init__(
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    def _predict_uncalibrated_coverage(
        self,
        coverage: float,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        samples = self.pls.predict_samples(
            x=x,
            particles=self.particles,
            predictive_noise=None,
            observation_noise=None,
        )
        lower_quantile, upper_quantile = 0.5 - coverage / 2, 0.5 + coverage / 2
        lower_bound = torch.quantile(samples, q=lower_quantile, dim=1)
        upper_bound = torch.quantile(samples, q=upper_quantile, dim=1)
        return lower_bound, upper_bound

    def predict_median(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        samples = self.pls.predict_samples(
            x=x,
            particles=self.particles,
            predictive_noise=None,
            observation_noise=None,
        )
        return torch.quantile(samples, q=0.5, dim=1)
