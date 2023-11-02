import math
import os
from typing import List, Union

import gpytorch
import pandas as pd
import torch

from experiments.data import ExperimentData
from src.conformalise import ConformaliseBase
from src.gps import ExactGP, svGP
from src.gradient_flows import ProjectedWassersteinGradientFlow
from src.temper import TemperBase


def calculate_mae(
    prediction: gpytorch.distributions.MultivariateNormal,
    y: torch.Tensor,
) -> float:
    return gpytorch.metrics.mean_absolute_error(
        pred_dist=prediction,
        test_y=y,
    ).item()


def calculate_nll(
    prediction: gpytorch.distributions.MultivariateNormal,
    y: torch.Tensor,
    y_std: float,
) -> float:
    return gpytorch.metrics.mean_standardized_log_loss(
        pred_dist=prediction,
        test_y=y,
    ).item() + math.log(y_std)


def calculate_metrics(
    model: Union[
        ConformaliseBase, ExactGP, svGP, TemperBase, ProjectedWassersteinGradientFlow
    ],
    experiment_data: ExperimentData,
    model_name: str,
    dataset_name: str,
    results_path: str,
):
    for data in [
        experiment_data.train,
        experiment_data.validation,
        experiment_data.test,
    ]:
        prediction = model(
            data.x  # not indicating parameter name because this can vary
        )
        mae = calculate_mae(
            y=data.y,
            prediction=prediction,
        )
        if not os.path.exists(os.path.join(results_path, model_name)):
            os.makedirs(os.path.join(results_path, model_name))
        pd.DataFrame([[mae]], columns=[model_name], index=[dataset_name]).to_csv(
            os.path.join(results_path, model_name, f"mae_{data.name}.csv"),
            index_label="dataset",
        )
        nll = calculate_nll(
            prediction=prediction,
            y=data.y,
            y_std=experiment_data.y_std,
        )
        pd.DataFrame([[nll]], columns=[model_name], index=[dataset_name]).to_csv(
            os.path.join(results_path, model_name, f"nll_{data.name}.csv"),
            index_label="dataset",
        )


def concatenate_metrics(
    results_path: str,
    data_types: List[str],
    models: List[str],
    datasets: List[str],
    metrics: List[str],
) -> None:
    for data_type in data_types:
        for metric in metrics:
            df_list = []
            for dataset in datasets:
                try:
                    df_list.append(
                        pd.concat(
                            [
                                pd.read_csv(
                                    os.path.join(
                                        results_path,
                                        dataset,
                                        model,
                                        f"{metric}_{data_type}.csv",
                                    ),
                                    index_col="dataset",
                                )
                                for model in models
                            ],
                            axis=1,
                        )
                    )
                except Exception as e:
                    print(e)
                    print(f"Dataset {dataset} failed to load results.")
            pd.concat(df_list, axis=0).to_csv(
                os.path.join(
                    results_path,
                    f"{metric}_{data_type}.csv",
                ),
                index_label="dataset",
            )
