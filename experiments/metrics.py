import math
import os
from typing import List, Optional, Union

import gpytorch
import pandas as pd
import torch

from experiments.data import ExperimentData
from experiments.plotters import plot_true_versus_predicted
from experiments.utils import create_directory
from src.gps import ExactGP, svGP
from src.gradient_flows.base.base import GradientFlowBase
from src.utils import set_seed


def calculate_mae(
    prediction: torch.distributions.Distribution,
    y: torch.Tensor,
) -> float:
    if isinstance(prediction, gpytorch.distributions.MultivariateNormal):
        return gpytorch.metrics.mean_absolute_error(
            pred_dist=prediction,
            test_y=y,
        ).item()
    elif isinstance(prediction, torch.distributions.Bernoulli):
        return prediction.probs.sub(y).abs().mean().item()
    else:
        raise ValueError(f"Prediction type {type(prediction)} not supported")


def calculate_mse(
    prediction: torch.distributions.Distribution,
    y: torch.Tensor,
) -> float:
    if isinstance(prediction, gpytorch.distributions.MultivariateNormal):
        return gpytorch.metrics.mean_squared_error(
            pred_dist=prediction,
            test_y=y,
        ).item()
    elif isinstance(prediction, torch.distributions.Bernoulli):
        return prediction.probs.sub(y).pow(2).mean().item()
    else:
        raise ValueError(f"Prediction type {type(prediction)} not supported")


def calculate_nll(
    prediction: torch.distributions.Distribution,
    y: torch.Tensor,
) -> float:
    if isinstance(prediction, gpytorch.distributions.MultivariateNormal):
        return gpytorch.metrics.mean_standardized_log_loss(
            pred_dist=prediction,
            test_y=y,
        ).item()
    elif isinstance(prediction, torch.distributions.Bernoulli):
        return torch.nn.functional.binary_cross_entropy(
            input=prediction.probs,
            target=y.double(),
            reduction="mean",
        ).item()
    else:
        raise ValueError(f"Prediction type {type(prediction)} not supported")


def calculate_metrics(
    model: Union[ExactGP, svGP, GradientFlowBase],
    experiment_data: ExperimentData,
    model_name: str,
    dataset_name: str,
    results_path: str,
    plots_path: str,
    particles: Optional[torch.Tensor] = None,
):
    for _model, _model_name in [
        (model, model_name),
    ]:
        for data in [
            experiment_data.train,
            experiment_data.test,
        ]:
            set_seed(0)
            if isinstance(_model, svGP) or isinstance(_model, ExactGP):
                prediction = _model.likelihood(_model(data.x))
            elif isinstance(_model, GradientFlowBase):
                prediction = _model(x=data.x, particles=particles)
            else:
                raise ValueError(f"Model type {type(_model)} not supported")
            mae = calculate_mae(
                y=data.y,
                prediction=prediction,
            )
            create_directory(os.path.join(results_path, _model_name))
            pd.DataFrame([[mae]], columns=[_model_name], index=[dataset_name]).to_csv(
                os.path.join(results_path, _model_name, f"mae_{data.name}.csv"),
                index_label="dataset",
            )
            mse = calculate_mse(
                y=data.y,
                prediction=prediction,
            )
            create_directory(os.path.join(results_path, _model_name))
            pd.DataFrame([[mse]], columns=[_model_name], index=[dataset_name]).to_csv(
                os.path.join(results_path, _model_name, f"mse_{data.name}.csv"),
                index_label="dataset",
            )
            nll = calculate_nll(
                prediction=prediction,
                y=data.y,
            )
            pd.DataFrame([[nll]], columns=[_model_name], index=[dataset_name]).to_csv(
                os.path.join(results_path, _model_name, f"nll_{data.name}.csv"),
                index_label="dataset",
            )
            create_directory(os.path.join(plots_path, _model_name))
            plot_true_versus_predicted(
                y_true=data.y,
                y_pred=prediction,
                title=f"True versus Predicted ({mae=:.2f},{mse=:.2f},{nll=:.2f}) ({dataset_name},{_model_name},{data.name} data)",
                save_path=os.path.join(
                    plots_path, _model_name, f"true_versus_predicted_{data.name}.png"
                ),
                error_bar=True,
            )


def concatenate_metrics(
    results_path: str,
    data_types: List[str],
    model_names: List[str],
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
                                for model in model_names
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
