import torch

from projected_langevin_sampling.utils import get_torch_generator


def _safe_eigh(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return torch.linalg.eigh(cov)
    except torch.linalg.LinAlgError:
        symmetric_cov = (cov + cov.mT) / 2
        try:
            return torch.linalg.eigh(symmetric_cov)
        except torch.linalg.LinAlgError:
            if not symmetric_cov.is_cuda:
                raise
            eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_cov.cpu())
            return (
                eigenvalues.to(device=cov.device, dtype=cov.dtype),
                eigenvectors.to(device=cov.device, dtype=cov.dtype),
            )


def sample_multivariate_normal(
    mean: torch.Tensor,
    cov: torch.Tensor,
    size: tuple[int, ...] | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Wrapper for pytorch multivariate normal sampler which removes negative eigenvalues as a work around for
    non-positive definite covariance matrices.

    :param mean: mean vector
    :param cov: covariance matrix
    :param size: output size
    :param seed: optional seed for sampler
    :return: multivariate normal samples
    """
    size = (1,) if not size else size
    eigenvalues, eigenvectors = _safe_eigh(cov)
    mean = mean.to(device=eigenvalues.device, dtype=eigenvalues.dtype)
    # Ensure that the eigenvalues are positive
    eigenvalues = torch.clip(eigenvalues, 0, None)
    normal_sample = torch.empty(
        (eigenvalues.shape[0], *size),
        device=eigenvalues.device,
        dtype=eigenvalues.dtype,
    ).normal_(
        mean=0.0,
        std=1.0,
        generator=get_torch_generator(seed=seed, device=eigenvalues.device),
    )
    return torch.real(
        mean[:, None]
        + eigenvectors @ torch.diag(torch.sqrt(eigenvalues)) @ normal_sample
    ).T


def sample_point(
    x: torch.Tensor,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Sample an item in a vector
    :param x: a vector
    :param seed: optional seed for sampler
    :return: an item sampled from the vector
    """
    random_idx = torch.randperm(
        x.shape[0],
        device=x.device,
        generator=get_torch_generator(seed=seed, device=x.device),
    )[0]
    return x[random_idx : random_idx + 1, ...]
