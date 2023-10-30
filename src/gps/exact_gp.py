import gpytorch
import torch


class ExactGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        mean: gpytorch.means.Mean,
        kernel: gpytorch.kernels.Kernel,
        x: torch.Tensor,
        y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
    ):
        super(ExactGP, self).__init__(x, y, likelihood)
        self.mean = mean
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            mean=self.mean(x),
            covariance_matrix=self.kernel(x),
        )
