import gpytorch
import torch


class PLSKernel(gpytorch.kernels.Kernel):
    """
    Wraps a base kernel with a projected Langevin sampling kernel such that
    K(x1, x2) = 1 / N * sum_{i=1}^N K_b(x1, z_i) * K_b(x2, z_i)
    where K_b is the base kernel and z_i are the approximation samples.
    This is defined as kernel r in the paper, while the base kernel is kernel
    k in the paper.
    """

    is_stationary: bool = False

    def __init__(
        self,
        base_kernel: gpytorch.kernels.Kernel,
        approximation_samples: torch.Tensor,
        **kwargs,
    ):
        super(PLSKernel, self).__init__(**kwargs)

        self.base_kernel = base_kernel
        self.approximation_samples = approximation_samples

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        additional_approximation_samples: torch.Tensor | None = None,
        last_dim_is_batch: bool = False,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        approximation_samples_list = [self.approximation_samples]
        if additional_approximation_samples is not None:
            approximation_samples_list.append(additional_approximation_samples)
        approximation_samples = torch.cat(approximation_samples_list, dim=0).unique(
            dim=0
        )
        number_of_approximation_samples = approximation_samples.shape[0]
        if torch.cuda.is_available():
            gram_x1_sample = self.base_kernel.cuda().forward(
                x1=x1,
                x2=approximation_samples,
                last_dim_is_batch=last_dim_is_batch,
            )
            gram_x2_sample = self.base_kernel.cuda().forward(
                x1=x2,
                x2=approximation_samples,
                last_dim_is_batch=last_dim_is_batch,
            )
        else:
            gram_x1_sample = self.base_kernel.forward(
                x1=x1,
                x2=approximation_samples,
                last_dim_is_batch=last_dim_is_batch,
            )
            gram_x2_sample = self.base_kernel.forward(
                x1=x2,
                x2=approximation_samples,
                last_dim_is_batch=last_dim_is_batch,
            )
        res = torch.mul(
            torch.div(1, number_of_approximation_samples),
            (gram_x1_sample @ gram_x2_sample.T),
        )
        if diag:
            return res.diag()
        else:
            return res

    def num_outputs_per_input(self, x1: torch.Tensor, x2: torch.Tensor) -> int:
        return 1
