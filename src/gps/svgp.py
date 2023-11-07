import gpytorch
import torch
from gpytorch.models import ApproximateGP


class svGP(ApproximateGP):
    def __init__(
        self,
        mean: gpytorch.means.Mean,
        kernel: gpytorch.kernels.Kernel,
        x_induce: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        learn_inducing_locations: bool = True,
    ):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            x_induce.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            x_induce,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super(svGP, self).__init__(variational_strategy)
        self.mean = mean
        self.kernel = kernel
        self.likelihood = likelihood

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            mean=self.mean(x),
            covariance_matrix=self.kernel(x),
        )
