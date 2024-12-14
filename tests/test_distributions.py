import numpy as np
import pytest
import torch

from src.distributions import StudentTMarginals


@pytest.mark.parametrize(
    "df,loc,scale,y,expected_log_likelihood",
    [
        [
            5.2,
            torch.tensor([-3.4, 1.6, 4.3]),
            torch.tensor([0.9, 9.1, 0.1]),
            torch.tensor([4, 3.1, 4]),
            4.6707,
        ],
    ],
)
def test_student_t(
    df: float,
    loc: torch.Tensor,
    scale: torch.Tensor,
    y: torch.Tensor,
    expected_log_likelihood: float,
):
    assert np.allclose(
        StudentTMarginals(
            df=df,
            loc=loc,
            scale=scale,
        ).negative_log_likelihood(y=y),
        expected_log_likelihood,
    )
