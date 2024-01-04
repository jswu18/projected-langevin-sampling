from typing import Tuple

import torch
from sklearn.model_selection import train_test_split

from experiments.data import Data, ExperimentData


def _split_regression_data_intervals(
    seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    interval_size = x.shape[0] // total_number_of_intervals
    number_of_intervals_on_edge = int(1 / 8 * total_number_of_intervals)
    test_interval_indices = (
        torch.randperm(
            total_number_of_intervals - 2 * number_of_intervals_on_edge,
            generator=torch.Generator().manual_seed(seed),
        )[:number_of_test_intervals]
        + number_of_intervals_on_edge
    )
    x_train = torch.concatenate(
        [
            x[interval_size * interval : interval_size * (interval + 1)]
            for interval in range(total_number_of_intervals)
            if interval not in test_interval_indices
        ]
    )
    y_train = torch.concatenate(
        [
            y[interval_size * interval : interval_size * (interval + 1)]
            for interval in range(total_number_of_intervals)
            if interval not in test_interval_indices
        ]
    )
    x_test = torch.concatenate(
        [
            x[interval_size * interval : interval_size * (interval + 1)]
            for interval in range(total_number_of_intervals)
            if interval in test_interval_indices
        ]
    )
    y_test = torch.concatenate(
        [
            y[interval_size * interval : interval_size * (interval + 1)]
            for interval in range(total_number_of_intervals)
            if interval in test_interval_indices
        ]
    )
    return x_train, y_train, x_test, y_test


def split_regression_data_intervals(
    seed: int,
    split_seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    train_data_percentage: float,
):
    (
        x_train_validation,
        y_train_validation,
        x_test,
        y_test,
    ) = _split_regression_data_intervals(
        seed=split_seed,
        x=x,
        y=y,
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=1 - train_data_percentage,
        random_state=seed,
    )
    return (
        x_train,
        y_train,
        x_test,
        y_test,
        x_validation,
        y_validation,
    )


def split_regression_data(
    seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    train_data_percentage: float,
    validation_data_percentage: float,
    test_data_percentage: float,
):
    (
        x_train,
        x_test_and_validation,
        y_train,
        y_test_and_validation,
    ) = train_test_split(
        x,
        y,
        test_size=1 - train_data_percentage,
        random_state=seed,
    )

    x_validation, x_test, y_validation, y_test = train_test_split(
        x_test_and_validation,
        y_test_and_validation,
        test_size=test_data_percentage
        / (test_data_percentage + validation_data_percentage),
        random_state=seed,
    )
    return (
        torch.tensor(x_train),
        torch.tensor(y_train),
        torch.tensor(x_test),
        torch.tensor(y_test),
        torch.tensor(x_validation),
        torch.tensor(y_validation),
    )


def set_up_experiment(
    name: str,
    seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    train_data_percentage: float,
    normalise: bool = True,
) -> ExperimentData:
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        x,
        y,
        test_size=1 - train_data_percentage,
        random_state=seed,
    )
    if normalise:
        y_mean = torch.mean(y_train)
        y_std = torch.std(y_train)
        y = (y - y_mean) / y_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
    else:
        y_mean = 0.0
        y_std = 1.0
    experiment_data = ExperimentData(
        name=name,
        full=Data(x=torch.tensor(x), y=torch.tensor(y), name="full"),
        train=Data(x=torch.tensor(x_train), y=torch.tensor(y_train), name="train"),
        test=Data(x=torch.tensor(x_test), y=torch.tensor(y_test), name="test"),
        y_mean=y_mean,
        y_std=y_std,
    )
    return experiment_data
