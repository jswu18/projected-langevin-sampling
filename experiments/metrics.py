import os
from typing import List

import gpytorch
import pandas as pd
import torch

from experiments.data import ExperimentData
from src.conformalise import ConformaliseBase, ConformaliseGP, ConformaliseGradientFlow
from src.gradient_flows import ProjectedWassersteinGradientFlow


def calculate_nll_particles(
    y: torch.Tensor,
    y_std: torch.Tensor,
    predicted_samples: torch.Tensor,
    jitter=1e-10,
):
    return gpytorch.metrics.mean_standardized_log_loss(
        gpytorch.distributions.MultivariateNormal(
            mean=predicted_samples.mean(axis=1),
            covariance_matrix=torch.diag(
                torch.clip(predicted_samples.var(axis=1), jitter, None)
            ),
        ),
        y,
    ) + torch.log(y_std)


def calculate_nll_conformal(
    y: torch.Tensor,
    y_std: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_variance: torch.Tensor,
    jitter=1e-10,
):
    return gpytorch.metrics.mean_standardized_log_loss(
        gpytorch.distributions.MultivariateNormal(
            mean=predicted_mean,
            covariance_matrix=torch.diag(torch.clip(predicted_variance, jitter, None)),
        ),
        y,
    ) + torch.log(y_std)


def calculate_mae(
    y: torch.Tensor,
    predicted: torch.Tensor,
):
    return torch.mean(torch.abs(y - predicted))


def calculate_particle_metrics(
    model: ProjectedWassersteinGradientFlow,
    model_name: str,
    dataset_name: str,
    particles: torch.Tensor,
    experiment_data: ExperimentData,
    results_path: str,
) -> None:
    if not os.path.exists(os.path.join(results_path, model_name)):
        os.makedirs(os.path.join(results_path, model_name))
    conformal_model = ConformaliseGradientFlow(
        x_calibration=experiment_data.validation.x,
        y_calibration=experiment_data.validation.y,
        gradient_flow=model,
        particles=particles,
    )
    _calculate_metrics(
        model=conformal_model,
        experiment_data=experiment_data,
        model_name=model_name,
        dataset_name=dataset_name,
        results_path=results_path,
    )


def calculate_svgp_metrics(
    model: gpytorch.models.VariationalGP,
    model_name: str,
    dataset_name: str,
    experiment_data: ExperimentData,
    results_path: str,
) -> None:
    if not os.path.exists(os.path.join(results_path, model_name)):
        os.makedirs(os.path.join(results_path, model_name))
    conformal_model = ConformaliseGP(
        x_calibration=experiment_data.validation.x,
        y_calibration=experiment_data.validation.y,
        gp=model,
    )
    _calculate_metrics(
        model=conformal_model,
        experiment_data=experiment_data,
        model_name=model_name,
        dataset_name=dataset_name,
        results_path=results_path,
    )


def _calculate_metrics(
    model: ConformaliseBase,
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
        mean = model.predict(
            x=data.x,
        )
        variance = model.predict_variance(
            x=data.x,
        )
        mae = calculate_mae(
            y=data.y,
            predicted=mean,
        )
        pd.DataFrame(
            [[float(mae.detach().item())]], columns=[model_name], index=[dataset_name]
        ).to_csv(
            f"{results_path}/{model_name}/mae_{data.name}.csv",
            index_label="dataset",
        )
        nll = calculate_nll_conformal(
            y=data.y,
            y_std=torch.tensor(experiment_data.y_std),
            predicted_mean=mean,
            predicted_variance=variance,
        )
        pd.DataFrame(
            [[float(nll.detach().item())]], columns=[model_name], index=[dataset_name]
        ).to_csv(
            f"{results_path}/{model_name}/nll_{data.name}.csv",
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
