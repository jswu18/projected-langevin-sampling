from typing import Tuple

import numpy as np
import pytest
import torch

from src.samplers import sample_multivariate_normal, sample_point


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
        ]
    ],
)
def test_sample_multivariate_normal(
    mean: torch.Tensor,
    cov: torch.Tensor,
    size: Tuple[int],
    seed: int,
    expected_sample: torch.Tensor,
):
    sample = sample_multivariate_normal(
        mean=mean,
        cov=cov,
        size=size,
        seed=seed,
    )
    assert np.allclose(sample, expected_sample)


@pytest.mark.parametrize(
    "x,seed,expected_sample",
    [[torch.tensor([1, 2, 3]), 0, torch.tensor([3])]],
)
def test_sample_point(
    x,
    seed: int,
    expected_sample: torch.Tensor,
):
    sample = sample_point(
        x=x,
        seed=seed,
    )
    assert np.allclose(sample, expected_sample)
