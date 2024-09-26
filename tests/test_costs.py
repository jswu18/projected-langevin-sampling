from typing import Any, Dict, Type

import gpytorch
import pytest
import torch

from src.projected_langevin_sampling.costs import (
    BernoulliCost,
    GaussianCost,
    PoissonCost,
)
from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import IdentityLinkFunction
from src.utils import set_seed


@pytest.mark.parametrize(
    "kwargs,CostType,prediction_samples,ExpectedDistributionType",
    [
        [
            {
                "y_train": torch.tensor([0.0, 1.0]),
            },
            BernoulliCost,
            torch.tensor([[0.4, 0.2], [0.3, 0.5]]),
            torch.distributions.Bernoulli,
        ],
        [
            {"y_train": torch.tensor([2.4, -2.3]), "observation_noise": 1.0},
            GaussianCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            gpytorch.distributions.MultivariateNormal,
        ],
        [
            {
                "y_train": torch.tensor([2.4, 2.3]),
            },
            PoissonCost,
            torch.tensor([[4.1, 3.2], [9.3, 2.5]]),
            torch.distributions.Poisson,
        ],
    ],
)
def test_cost_predict(
    kwargs: Dict[str, Any],
    CostType: Type[PLSCost],
    prediction_samples: torch.Tensor,
    ExpectedDistributionType: Type[torch.distributions.Distribution],
):
    cost = CostType(link_function=IdentityLinkFunction(), **kwargs)
    distribution = cost.predict(prediction_samples=prediction_samples)
    assert isinstance(distribution, ExpectedDistributionType)


@pytest.mark.parametrize(
    "kwargs,CostType,untransformed_train_prediction_samples,expected_cost",
    [
        [
            {
                "y_train": torch.tensor([0.0, 1.0]),
            },
            BernoulliCost,
            torch.tensor([[0.1, 0.2], [0.9, 0.5]]).double(),
            torch.tensor([0.21072109043598175, 0.9162907004356384]).double(),
        ],
        [
            {"y_train": torch.tensor([2.4, -2.3]), "observation_noise": 1.0},
            GaussianCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([[25.9450, 11.8400]]),
        ],
        [
            {
                "y_train": torch.tensor([2.4, 2.3]),
            },
            PoissonCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([-22.2308, -4.0981]),
        ],
    ],
)
def test_calculate_cost(
    kwargs: Dict[str, Any],
    CostType: Type[PLSCost],
    untransformed_train_prediction_samples: torch.Tensor,
    expected_cost: torch.Tensor,
):
    cost = CostType(link_function=IdentityLinkFunction(), **kwargs)
    cost = cost.calculate_cost(
        untransformed_train_prediction_samples=untransformed_train_prediction_samples
    )
    assert torch.allclose(cost, expected_cost)


# @pytest.mark.parametrize(
#     "kwargs,CostType,untransformed_train_prediction_samples,expected_cost_derivative",
#     [
#         [
#             {
#                 "y_train": torch.tensor([0.0, 1.0]),
#             },
#             BernoulliCost,
#             torch.tensor([[0.1, 0.2], [0.9, 0.5]]),
#             torch.tensor([[-0.1707, -0.5000], [1.4946, -0.8400]]),
#         ],
#         [
#             {"y_train": torch.tensor([2.4, -2.3]), "observation_noise": 1.0},
#             GaussianCost,
#             torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
#             torch.tensor([[1.7000,  0.8000], [-7.0000,  4.8000]]),
#         ],
#         [
#             {
#                 "y_train": torch.tensor([2.4, 2.3]),
#             },
#             PoissonCost,
#             torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
#             torch.tensor([[-0.1707, -0.5000], [1.4946, -0.8400]]),
#         ],
#     ],
# )
# def test_calculate_cost_derivative(
#     kwargs: Dict[str, Any],
#     CostType: Type[PLSCost],
#     untransformed_train_prediction_samples: torch.Tensor,
#     expected_cost_derivative: torch.Tensor,
# ):
#     set_seed(0)
#     cost = CostType(link_function=IdentityLinkFunction(), **kwargs)
#     cost_derivative = cost.calculate_cost_derivative(
#         untransformed_train_prediction_samples=untransformed_train_prediction_samples
#     )
#     print(cost_derivative)
#     assert torch.allclose(cost_derivative, expected_cost_derivative)
