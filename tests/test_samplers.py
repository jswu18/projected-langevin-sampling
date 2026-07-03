from typing import Tuple

import numpy as np
import pytest
import torch

from projected_langevin_sampling.samplers import (
    sample_multivariate_normal,
    sample_point,
)
from projected_langevin_sampling.utils import set_seed


@pytest.mark.parametrize(
    "mean,cov,size,seed,expected_sample",
    [
        [
            torch.zeros((2,)),
            torch.eye(2),
            (2,),
            0,
            torch.tensor(
                [
                    [1.5409960746765137, -2.1787893772125244],
                    [-0.293428897857666, 0.5684312582015991],
                ]
            ),
        ],
        [
            torch.zeros((2,)),
            torch.eye(2),
            (2,),
            None,
            torch.tensor(
                [
                    [1.5409960746765137, -2.1787893772125244],
                    [-0.293428897857666, 0.5684312582015991],
                ]
            ),
        ],
        [
            torch.zeros((2,)),
            torch.eye(2),
            None,
            None,
            torch.tensor(
                [
                    [1.5410, -0.2934],
                ]
            ),
        ],
    ],
)
def test_sample_multivariate_normal(
    mean: torch.Tensor,
    cov: torch.Tensor,
    size: Tuple[int],
    seed: int,
    expected_sample: torch.Tensor,
):
    set_seed(0)
    sample = sample_multivariate_normal(
        mean=mean,
        cov=cov,
        size=size,
        seed=seed,
    )
    assert np.allclose(sample, expected_sample, rtol=1e-3)


def test_sample_multivariate_normal_retries_eigh_failures(monkeypatch) -> None:
    original_eigh = torch.linalg.eigh
    call_count = 0

    def failing_once(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise torch.linalg.LinAlgError("forced failure")
        return original_eigh(cov)

    monkeypatch.setattr(torch.linalg, "eigh", failing_once)

    sample = sample_multivariate_normal(
        mean=torch.zeros((2,)),
        cov=torch.eye(2),
        size=(2,),
        seed=0,
    )

    assert call_count == 2
    assert sample.shape == (2, 2)


@pytest.mark.parametrize(
    "x,seed,expected_sample",
    [
        [torch.tensor([1, 2, 3]), 0, torch.tensor([3])],
        [torch.tensor([1, 2, 3]), None, torch.tensor([3])],
    ],
)
def test_sample_point(
    x,
    seed: int,
    expected_sample: torch.Tensor,
):
    set_seed(0)
    sample = sample_point(
        x=x,
        seed=seed,
    )
    assert np.allclose(sample, expected_sample)
