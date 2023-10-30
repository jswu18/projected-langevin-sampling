from typing import Tuple

import torch


def sample_multivariate_normal(
    mean: torch.Tensor,
    cov: torch.Tensor,
    size: Tuple[int] = None,
    seed: int = None,
) -> torch.Tensor:
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
    ).double()
    return torch.real(
        mean[:, None]
        + eigenvectors.double()
        @ torch.diag(torch.sqrt(eigenvalues)).double()
        @ normal_sample
    ).T
