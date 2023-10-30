import pytest
import torch

from mockers.kernel import MockKernel
from src.induce_data_selectors import (
    ConditionalVarianceInduceDataSelector,
    RandomInduceDataSelector,
)
from src.utils import set_seed


@pytest.mark.parametrize(
    "x,m,seed,z",
    [
        [
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.5, 2.5, 3.5],
                ]
            ),
            1,
            0,
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                ]
            ),
        ],
        [
            torch.tensor(
                [
                    [1.0, 3.0],
                    [3.0, 5.0],
                    [1.1, 3.5],
                    [1.3, 7.5],
                    [2.5, 2.5],
                ]
            ),
            1,
            0,
            torch.tensor(
                [
                    [2.5, 2.5],
                ]
            ),
        ],
    ],
)
def test_random_induce_data_selector(
    x: torch.Tensor,
    m: int,
    seed: int,
    z: torch.Tensor,
):
    kernel = MockKernel()
    set_seed(seed=seed)
    selector = RandomInduceDataSelector()
    z_computed, _ = selector.compute_induce_data(x=x, m=m, kernel=kernel)
    assert torch.allclose(z_computed, z)


@pytest.mark.parametrize(
    "x,m,seed,z",
    [
        [
            torch.tensor(
                [
                    [1.1, 3.5, 3.5],
                    [1.3, 7.5, 1.5],
                    [2.5, 2.5, 0.5],
                    [1.5, 2.5, 3.5],
                ]
            ),
            2,
            0,
            torch.tensor(
                [
                    [1.3, 7.5, 1.5],
                    [1.5, 2.5, 3.5],
                ]
            ),
        ],
        [
            torch.tensor(
                [
                    [1.0, 3.0],
                    [3.0, 5.0],
                    [1.1, 3.5],
                    [1.3, 7.5],
                    [2.5, 2.5],
                ]
            ),
            2,
            0,
            torch.tensor(
                [
                    [1.3, 7.5],
                    [3.0, 5.0],
                ]
            ),
        ],
    ],
)
def test_conditional_variance_induce_data_selector(
    x: torch.Tensor,
    m: int,
    seed: int,
    z: torch.Tensor,
):
    kernel = MockKernel()
    set_seed(seed=seed)
    selector = ConditionalVarianceInduceDataSelector()
    z_computed, _ = selector.compute_induce_data(x=x, m=m, kernel=kernel)
    assert torch.allclose(z_computed, z)
