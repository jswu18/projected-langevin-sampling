from typing import Any, Dict, Type

import gpytorch
import pytest
import torch

from src.distributions import StudentTMarginals
from src.projected_langevin_sampling.costs import (
    BernoulliCost,
    GaussianCost,
    MultiModalCost,
    PoissonCost,
)
from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.costs.student_t import StudentTCost
from src.projected_langevin_sampling.link_functions import (
    IdentityLinkFunction,
    ProbitLinkFunction,
    SigmoidLinkFunction,
    SquareLinkFunction,
)


@pytest.mark.parametrize(
    "kwargs,CostType,prediction_samples,ExpectedDistributionType",
    [
        [
            {
                "link_function": SigmoidLinkFunction(),
                "y_train": torch.tensor([0.0, 1.0]),
            },
            BernoulliCost,
            torch.tensor([[0.4, 0.2], [0.3, 0.5]]),
            torch.distributions.Bernoulli,
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, -2.3]),
                "observation_noise": 1.0,
            },
            GaussianCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            gpytorch.distributions.MultivariateNormal,
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, 2.3]),
                "degrees_of_freedom": 3,
            },
            StudentTCost,
            torch.tensor([[4.1, 3.2], [9.3, 2.5]]),
            StudentTMarginals,
        ],
        [
            {
                "link_function": SquareLinkFunction(),
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
    cost = CostType(**kwargs)
    distribution = cost.predict(prediction_samples=prediction_samples)
    assert isinstance(distribution, ExpectedDistributionType)


@pytest.mark.parametrize(
    "kwargs,CostType,untransformed_train_prediction_samples,expected_cost",
    [
        [
            {
                "link_function": SigmoidLinkFunction(),
                "y_train": torch.tensor([0.0, 1.0]),
            },
            BernoulliCost,
            torch.tensor([[0.1, 0.2], [0.9, 0.5]]).double(),
            torch.tensor([1.0856, 1.2722]).double(),
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, -2.3]),
                "observation_noise": 1.0,
            },
            GaussianCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([[25.9450, 11.8400]]),
        ],
        [
            {
                "link_function": SquareLinkFunction(),
                "y_train": torch.tensor([2.4, 2.3]),
            },
            PoissonCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([86.2692, 6.6919]),
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, 2.3]),
                "degrees_of_freedom": 3,
            },
            StudentTCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([9.0002, 0.4132]),
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, 2.3]),
                "observation_noise": 1.0,
                "shift": 15.8,
                "bernoulli_noise": 0.8,
            },
            MultiModalCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([73.7818, 5.3968]),
        ],
    ],
)
def test_calculate_cost(
    kwargs: Dict[str, Any],
    CostType: Type[PLSCost],
    untransformed_train_prediction_samples: torch.Tensor,
    expected_cost: torch.Tensor,
):
    cost = CostType(**kwargs)
    calculated_cost = cost.calculate_cost(
        untransformed_train_prediction_samples=untransformed_train_prediction_samples
    )
    assert torch.allclose(calculated_cost, expected_cost, rtol=1e-3)


@pytest.mark.parametrize(
    "kwargs,CostType,untransformed_train_prediction_samples,expected_cost_derivative",
    [
        [
            {
                "link_function": SigmoidLinkFunction(),
                "y_train": torch.tensor([0.0, 1.0]),
            },
            BernoulliCost,
            torch.tensor([[0.1, 0.2], [0.9, 0.5]]).double(),
            torch.tensor([[0.5250, 0.5498], [-0.2891, -0.3775]]).double(),
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, -2.3]),
                "observation_noise": 1.0,
            },
            GaussianCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([[1.7000, 0.8000], [-7.0000, 4.8000]]),
        ],
        [
            {
                "link_function": SquareLinkFunction(),
                "y_train": torch.tensor([2.4, 2.3]),
            },
            PoissonCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([[7.0293, 4.9000], [-18.1054, 3.1600]]),
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, 2.3]),
                "degrees_of_freedom": 3,
            },
            StudentTCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([[1.1545, 0.8791], [-0.3373, 0.2632]]),
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, 2.3]),
                "observation_noise": 1.0,
                "shift": 15.8,
                "bernoulli_noise": 0.8,
            },
            MultiModalCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([[1.7000, 0.8000], [-11.6000, 0.2000]]),
        ],
    ],
)
def test_calculate_cost_derivative(
    kwargs: Dict[str, Any],
    CostType: Type[PLSCost],
    untransformed_train_prediction_samples: torch.Tensor,
    expected_cost_derivative: torch.Tensor,
):
    cost = CostType(**kwargs)
    calculated_cost_derivative = cost.calculate_cost_derivative(
        untransformed_train_prediction_samples=untransformed_train_prediction_samples
    )
    assert torch.allclose(
        calculated_cost_derivative, expected_cost_derivative, rtol=1e-3
    )


@pytest.mark.parametrize(
    "kwargs,CostType,untransformed_train_prediction_samples,expected_cost_derivative",
    [
        [
            {
                "link_function": ProbitLinkFunction(),
                "y_train": torch.tensor([0.0, 1.0]),
            },
            BernoulliCost,
            torch.tensor([[0.1, 0.2], [0.9, 0.5]]).double(),
            torch.tensor([[0.8626, 0.9294], [-0.3261, -0.5092]]).double(),
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, -2.3]),
                "observation_noise": 1.0,
            },
            GaussianCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([[1.7000, 0.8000], [-7.0000, 4.8000]]),
        ],
        [
            {
                "link_function": IdentityLinkFunction(),
                "y_train": torch.tensor([2.4, 2.3]),
            },
            PoissonCost,
            torch.tensor([[4.1, 3.2], [-9.3, 2.5]]),
            torch.tensor([[-0.1707, -0.5000], [1.4946, -0.8400]]),
        ],
    ],
)
def test_calculate_autograd_cost_derivative(
    kwargs: Dict[str, Any],
    CostType: Type[PLSCost],
    untransformed_train_prediction_samples: torch.Tensor,
    expected_cost_derivative: torch.Tensor,
):
    cost = CostType(**kwargs)
    calculated_cost_derivative = cost.calculate_cost_derivative(
        untransformed_train_prediction_samples=untransformed_train_prediction_samples,
        force_autograd=True,
    )
    assert torch.allclose(
        calculated_cost_derivative, expected_cost_derivative, rtol=1e-3
    )
