import os
from typing import List

import gpytorch
import pandas as pd
import torch

from experiments.data import ExperimentData
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
    mae_train = calculate_mae(
        y=experiment_data.train.y,
        predicted=model.predict(
            x=experiment_data.train.x,
            particles=particles,
        ).mean(axis=1),
    )
    pd.DataFrame(
        [[float(mae_train.detach().item())]], columns=[model_name], index=[dataset_name]
    ).to_csv(
        f"{results_path}/{model_name}/mae_train.csv",
        index_label="dataset",
    )
    mae_validation = calculate_mae(
        y=experiment_data.validation.y,
        predicted=model.predict(
            x=experiment_data.validation.x,
            particles=particles,
        ).mean(axis=1),
    )
    pd.DataFrame(
        [[float(mae_validation.detach().item())]],
        columns=[model_name],
        index=[dataset_name],
    ).to_csv(
        f"{results_path}/{model_name}/mae_validation.csv",
        index_label="dataset",
    )
    mae_test = calculate_mae(
        y=experiment_data.test.y,
        predicted=model.predict(
            x=experiment_data.test.x,
            particles=particles,
        ).mean(axis=1),
    )
    pd.DataFrame(
        [[float(mae_test.detach().item())]], columns=[model_name], index=[dataset_name]
    ).to_csv(
        f"{results_path}/{model_name}/mae_test.csv",
        index_label="dataset",
    )
    nll_train = calculate_nll_particles(
        y=experiment_data.train.y,
        y_std=torch.tensor(experiment_data.y_std),
        predicted_samples=model.predict(
            x=experiment_data.train.x,
            particles=particles,
        ),
    )
    pd.DataFrame(
        [[float(nll_train.detach().item())]], columns=[model_name], index=[dataset_name]
    ).to_csv(
        f"{results_path}/{model_name}/nll_train.csv",
        index_label="dataset",
    )
    nll_validation = calculate_nll_particles(
        y=experiment_data.validation.y,
        y_std=torch.tensor(experiment_data.y_std),
        predicted_samples=model.predict(
            x=experiment_data.validation.x,
            particles=particles,
        ),
    )
    pd.DataFrame(
        [[float(nll_validation.detach().item())]],
        columns=[model_name],
        index=[dataset_name],
    ).to_csv(
        f"{results_path}/{model_name}/nll_validation.csv",
        index_label="dataset",
    )
    nll_test = calculate_nll_particles(
        y=experiment_data.test.y,
        y_std=torch.tensor(experiment_data.y_std),
        predicted_samples=model.predict(
            x=experiment_data.test.x,
            particles=particles,
        ),
    )
    pd.DataFrame(
        [[float(nll_test.detach().item())]], columns=[model_name], index=[dataset_name]
    ).to_csv(
        f"{results_path}/{model_name}/nll_test.csv",
        index_label="dataset",
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
    mae_train = calculate_mae(
        y=experiment_data.train.y,
        predicted=model(
            experiment_data.train.x,
        ).mean,
    )
    pd.DataFrame(
        [[float(mae_train.detach().item())]], columns=[model_name], index=[dataset_name]
    ).to_csv(
        f"{results_path}/{model_name}/mae_train.csv",
        index_label="dataset",
    )
    mae_validation = calculate_mae(
        y=experiment_data.validation.y,
        predicted=model(
            experiment_data.validation.x,
        ).mean,
    )
    pd.DataFrame(
        [[float(mae_validation.detach().item())]],
        columns=[model_name],
        index=[dataset_name],
    ).to_csv(
        f"{results_path}/{model_name}/mae_validation.csv",
        index_label="dataset",
    )
    mae_test = calculate_mae(
        y=experiment_data.test.y,
        predicted=model(
            experiment_data.test.x,
        ).mean,
    )
    pd.DataFrame(
        [[float(mae_test.detach().item())]], columns=[model_name], index=[dataset_name]
    ).to_csv(
        f"{results_path}/{model_name}/mae_test.csv",
        index_label="dataset",
    )
    nll_train = gpytorch.metrics.mean_standardized_log_loss(
        model(experiment_data.train.x),
        experiment_data.train.y,
    ) + torch.log(torch.tensor(experiment_data.y_std))
    pd.DataFrame(
        [[float(nll_train.detach().item())]], columns=[model_name], index=[dataset_name]
    ).to_csv(
        f"{results_path}/{model_name}/nll_train.csv",
        index_label="dataset",
    )
    nll_validation = gpytorch.metrics.mean_standardized_log_loss(
        model(experiment_data.validation.x),
        experiment_data.validation.y,
    ) + torch.log(torch.tensor(experiment_data.y_std))
    pd.DataFrame(
        [[float(nll_validation.detach().item())]],
        columns=[model_name],
        index=[dataset_name],
    ).to_csv(
        f"{results_path}/{model_name}/nll_validation.csv",
        index_label="dataset",
    )
    nll_test = gpytorch.metrics.mean_standardized_log_loss(
        model(experiment_data.test.x),
        experiment_data.test.y,
    ) + torch.log(torch.tensor(experiment_data.y_std))
    pd.DataFrame(
        [[float(nll_test.detach().item())]], columns=[model_name], index=[dataset_name]
    ).to_csv(
        f"{results_path}/{model_name}/nll_test.csv",
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
