import pytest
import torch

from src.utils import set_seed


@pytest.mark.parametrize(
    "seed, expected_int",
    [[0, 4], [1, 5]],
)
def test_set_seed(
    seed: int,
    expected_int: int,
):
    set_seed(seed)
    assert torch.randint(low=0, high=10, size=(1,)).item() == expected_int
