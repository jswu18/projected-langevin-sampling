import math
import os
from typing import List, Union

import gpytorch
import pandas as pd
import torch

from experiments.constructors import (
    construct_conformalised_model,
    construct_tempered_model,
)
from experiments.data import ExperimentData
from experiments.plotters import plot_true_versus_predicted
from src.conformalise import ConformaliseBase
from src.gps import ExactGP, svGP
from src.gradient_flows import ProjectedWassersteinGradientFlow
from src.temper import TemperBase
from src.utils import set_seed


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
    model: Union[ExactGP, svGP, ProjectedWassersteinGradientFlow],
    experiment_data: ExperimentData,
    model_name: str,
    dataset_name: str,
    results_path: str,
    plots_path: str,
):
    temper_model = construct_tempered_model(
        model=model,
        data=experiment_data.validation,
    )
    conformal_model = construct_conformalised_model(
        model=model,
        data=experiment_data.validation,
    )
    for _model, _model_name in [
        (model, model_name),
        (temper_model, f"{model_name}-temper"),
        (conformal_model, f"{model_name}-conformal"),
    ]:
        for data in [
            experiment_data.train,
            experiment_data.validation,
            experiment_data.test,
        ]:
            set_seed(0)
            prediction = _model(
                data.x  # not indicating parameter name because this can vary
            )
            if isinstance(_model, svGP):
                prediction = _model.likelihood(prediction)
            mae = calculate_mae(
                y=data.y,
                prediction=prediction,
            )
            if not os.path.exists(os.path.join(results_path, _model_name)):
                os.makedirs(os.path.join(results_path, _model_name))
            pd.DataFrame([[mae]], columns=[_model_name], index=[dataset_name]).to_csv(
                os.path.join(results_path, _model_name, f"mae_{data.name}.csv"),
                index_label="dataset",
            )
            nll = calculate_nll(
                prediction=prediction,
                y=data.y,
                y_std=experiment_data.y_std,
            )
            pd.DataFrame([[nll]], columns=[_model_name], index=[dataset_name]).to_csv(
                os.path.join(results_path, _model_name, f"nll_{data.name}.csv"),
                index_label="dataset",
            )
            if not os.path.exists(os.path.join(plots_path, _model_name)):
                os.makedirs(os.path.join(plots_path, _model_name))
            plot_true_versus_predicted(
                y_true=data.y,
                y_pred=prediction,
                title=f"True versus Predicted ({mae=:.2f},{nll=:.2f}) ({dataset_name},{_model_name},{data.name} data)",
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
    model_names_ = []
    for model_type in ["", "-temper", "-conformal"]:
        model_names_.extend([f"{model_name}{model_type}" for model_name in model_names])
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
                                for model in model_names_
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
