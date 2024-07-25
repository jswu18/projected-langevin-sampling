import numpy as np
import pytest

from src.bisection_search import LogBisectionSearch


@pytest.mark.parametrize(
    "lower,upper,soft_update,expected",
    [
        [1.0e-1, 1.0e1, False, 1.0],
        [1.0e-1, 1.0e1, True, 1.0],
    ],
)
def test_bisection_search_current(
    lower: float,
    upper: float,
    soft_update: bool,
    expected: float,
):
    search = LogBisectionSearch(
        lower=lower,
        upper=upper,
        soft_update=soft_update,
    )
    assert np.isclose(search.current, expected)


@pytest.mark.parametrize(
    "lower,upper,soft_update,expected",
    [
        [
            1.0e-1,
            1.0e1,
            False,
            0.316227766016838,
        ],
        [
            1.0e-1,
            1.0e1,
            True,
            0.5623413251903492,
        ],
    ],
)
def test_bisection_search_update_upper(
    lower: float,
    upper: float,
    soft_update: bool,
    expected: float,
):
    search = LogBisectionSearch(
        lower=lower,
        upper=upper,
        soft_update=soft_update,
    )
    search.update_upper()
    assert np.isclose(search.current, expected)


@pytest.mark.parametrize(
    "lower,upper,soft_update,expected",
    [
        [
            1.0e-1,
            1.0e1,
            False,
            3.1622776601683795,
        ],
        [
            1.0e-1,
            1.0e1,
            True,
            1.7782794100389232,
        ],
    ],
)
def test_bisection_search_update_lower(
    lower: float,
    upper: float,
    soft_update: bool,
    expected: float,
):
    search = LogBisectionSearch(
        lower=lower,
        upper=upper,
        soft_update=soft_update,
    )
    search.update_lower()
    assert np.isclose(search.current, expected)
