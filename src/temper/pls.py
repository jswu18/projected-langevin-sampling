import gpytorch
import torch

from src.projected_langevin_sampling import PLS
from src.projected_langevin_sampling.costs import GaussianCost
from src.temper.base import TemperBase


class TemperPLS(TemperBase):
    """
    Temper the predictive variance of a Projected Langevin Sampling model.
    The model must use a Gaussian cost function, implying a regression task.
    """

    def __init__(
        self,
        x_calibration: torch.Tensor,
        y_calibration: torch.Tensor,
        pls: PLS,
        particles: torch.Tensor,
        debug: bool = False,
    ):
        self.debug = debug
        if not self.debug:
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
        """
        Predict the untempered predictive distribution. This is the predictive
        distribution of the Projected Langevin Sampling model.
        :param x: Input data of shape (N, D).
        :return: The predictive distribution.
        """
        prediction_distribution = self.pls(
            x=x,
            particles=self.particles,
            predictive_noise=None,
            observation_noise=None,
        )
        if not self.debug:
            assert isinstance(
                prediction_distribution, gpytorch.distributions.MultivariateNormal
            )
        return prediction_distribution
