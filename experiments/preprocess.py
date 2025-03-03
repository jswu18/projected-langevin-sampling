from typing import Optional, Tuple

import torch
from sklearn.model_selection import train_test_split

from experiments.data import Data, ExperimentData, ProblemType


def _split_regression_data_intervals(
    seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    y_untransformed: torch.Tensor | None = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    interval_size = x.shape[0] // total_number_of_intervals
    number_of_intervals_on_edge = max(int(1 / 8 * total_number_of_intervals), 3)
    test_interval_indices = list(
        torch.randperm(
            total_number_of_intervals - 2 * number_of_intervals_on_edge,
            generator=torch.Generator().manual_seed(seed),
        )[:number_of_test_intervals]
        + number_of_intervals_on_edge
    ) + [0, 1, total_number_of_intervals - 2, total_number_of_intervals - 1]
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
    if y_untransformed is not None:
        y_train_untransformed = torch.concatenate(
            [
                y_untransformed[
                    interval_size * interval : interval_size * (interval + 1)
                ]
                for interval in range(total_number_of_intervals)
                if interval not in test_interval_indices
            ]
        )
    else:
        y_train_untransformed = None
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
    if y_untransformed is not None:
        y_test_untransformed = torch.concatenate(
            [
                y_untransformed[
                    interval_size * interval : interval_size * (interval + 1)
                ]
                for interval in range(total_number_of_intervals)
                if interval in test_interval_indices
            ]
        )
    else:
        y_test_untransformed = None
    return x_train, y_train, y_train_untransformed, x_test, y_test, y_test_untransformed


def split_regression_data_intervals(
    split_seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    y_untransformed: torch.Tensor | None = None,
):
    (
        x_train,
        y_train,
        y_train_untransformed,
        x_test,
        y_test,
        y_test_untransformed,
    ) = _split_regression_data_intervals(
        seed=split_seed,
        x=x,
        y=y,
        y_untransformed=y_untransformed,
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
    )
    return (
        x_train,
        y_train,
        y_train_untransformed,
        x_test,
        y_test,
        y_test_untransformed,
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
    problem_type: ProblemType,
    seed: int,
    x: torch.Tensor,
    y: torch.Tensor,
    train_data_percentage: float,
    normalise: bool = True,
    validation_data_percentage: Optional[float] = 0.0,
) -> ExperimentData:
    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        x,
        y,
        test_size=1 - (train_data_percentage + validation_data_percentage),
        random_state=seed,
    )
    if validation_data_percentage > 0:
        (
            x_train,
            x_validation,
            y_train,
            y_validation,
        ) = train_test_split(
            x_train,
            y_train,
            test_size=validation_data_percentage
            / (validation_data_percentage + train_data_percentage),
            random_state=seed,
        )
    else:
        x_validation, y_validation = None, None
    if normalise:
        y_mean = torch.mean(y_train)
        y_std = torch.std(y_train)
        y = (y - y_mean) / y_std
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
        if y_validation is not None:
            y_validation = (y_validation - y_mean) / y_std
    else:
        y_mean = 0.0
        y_std = 1.0
    experiment_data = ExperimentData(
        name=name,
        problem_type=problem_type,
        full=Data(x=torch.tensor(x), y=torch.tensor(y), name="full"),
        train=Data(x=torch.tensor(x_train), y=torch.tensor(y_train), name="train"),
        validation=Data(
            x=torch.tensor(x_validation),
            y=torch.tensor(y_validation),
            name="validation",
        )
        if validation_data_percentage > 0
        else None,
        test=Data(x=torch.tensor(x_test), y=torch.tensor(y_test), name="test"),
        y_mean=y_mean,
        y_std=y_std,
    )
    return experiment_data
