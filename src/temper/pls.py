import gpytorch
import torch

from src.projected_langevin_sampling import ProjectedLangevinSampling
from src.projected_langevin_sampling.costs import GaussianCost
from src.temper.base import TemperBase


class TemperGradientFlow(TemperBase):
    def __init__(
        self,
        x_calibration: torch.Tensor,
        y_calibration: torch.Tensor,
        pls: ProjectedLangevinSampling,
        particles: torch.Tensor,
    ):
        assert isinstance(pls.cost, GaussianCost)
        self.pls = pls
        self.particles = particles
        super().__init__(
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )

    def _untempered_predict(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        prediction_distribution = self.pls(
            x=x,
            particles=self.particles,
            predictive_noise=None,
            observation_noise=None,
        )
        assert isinstance(
            prediction_distribution, gpytorch.distributions.MultivariateNormal
        )
        return prediction_distribution
