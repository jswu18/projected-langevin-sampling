import gpytorch
import torch


class GradientFlowKernel(gpytorch.kernels.Kernel):
    """
    Decorates a base kernel with a gradient flow kernel such that
    K(x1, x2) = 1 / N * sum_{i=1}^N K_b(x1, z_i) * K_b(x2, z_i)
    where K_b is the base kernel and z_i are the approximation samples.
    """

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        approximation_samples: torch.Tensor,
        **kwargs,
    ):
        if base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(GradientFlowKernel, self).__init__(**kwargs)

        self.base_kernel = base_kernel
        self.approximation_samples = approximation_samples
        self.number_of_approximation_samples = approximation_samples.shape[0]

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        last_dim_is_batch: bool = False,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        gram_x1_sample = self.base_kernel.forward(
            x1=x1,
            x2=self.approximation_samples,
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
        )
        gram_x2_sample = self.base_kernel.forward(
            x1=x2,
            x2=self.approximation_samples,
            diag=diag,
            last_dim_is_batch=last_dim_is_batch,
        )
        return torch.mul(
            torch.div(1, self.number_of_approximation_samples),
            (gram_x1_sample @ gram_x2_sample.T),
        )

    def num_outputs_per_input(self, x1: torch.Tensor, x2: torch.Tensor) -> int:
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(
        self, train_inputs, train_prior_dist, train_labels, likelihood
    ):
        return self.base_kernel.prediction_strategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )
