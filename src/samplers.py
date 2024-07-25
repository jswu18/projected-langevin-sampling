from typing import Tuple

import torch


def sample_multivariate_normal(
    mean: torch.Tensor,
    cov: torch.Tensor,
    size: Tuple[int] | None = None,
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
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    size = (1,) if not size else size
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # Ensure that the eigenvalues are positive
    eigenvalues = torch.clip(eigenvalues, 0, None)
    normal_sample = torch.normal(
        mean=0.0,
        std=1.0,
        size=(eigenvalues.shape[0], *size),
        generator=generator,
    )
    if torch.cuda.is_available():
        eigenvalues = eigenvalues.cuda()
        eigenvectors = eigenvectors.cuda()
        normal_sample = normal_sample.cuda()
        mean = mean.cuda()
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
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None
    random_idx = torch.randperm(x.shape[0], generator=generator)[0]
    return x[random_idx : random_idx + 1, ...]
