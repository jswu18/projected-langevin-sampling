import gpytorch
import torch
from gpytorch.models import ApproximateGP


class SVGP(ApproximateGP):
    """
    A sparse variational Gaussian Process model following:
    https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
    """

    def __init__(
        self,
        mean: gpytorch.means.Mean,
        kernel: gpytorch.kernels.Kernel,
        x_induce: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        learn_inducing_locations: bool = False,
    ):
        """
        Constructor for the sparse variational Gaussian Process model.
        :param mean: The GP mean function.
        :param kernel: The GP kernel function.
        :param x_induce: The inducing points of shape (M, D).
        :param likelihood: The GP likelihood function.
        :param learn_inducing_locations: Whether to learn the inducing points.
        """
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            x_induce.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            x_induce,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super(SVGP, self).__init__(variational_strategy)
        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood
        if torch.cuda.is_available():
            self.cuda()
            self.likelihood.cuda()

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.Distribution:
        return gpytorch.distributions.MultivariateNormal(
            mean=torch.Tensor(self.mean(x)),
            covariance_matrix=self.kernel(x),
        )
