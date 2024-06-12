import gpytorch
import torch


class ExactGP(gpytorch.models.ExactGP):
    """
    An exact Gaussian Process model following:
    https://docs.gpytorch.ai/en/v1.6.0/examples/01_Exact_GPs/Simple_GP_Regression.html
    """

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
        if torch.cuda.is_available():
            self.cuda()
            self.likelihood.cuda()

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            mean=self.mean(x),
            covariance_matrix=self.kernel(x),
        )
