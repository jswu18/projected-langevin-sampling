import os
from typing import List, Union

import gpytorch
import pandas as pd
import scipy
import sklearn
import torch

from experiments.data import ExperimentData
from experiments.plotters import plot_true_versus_predicted
from experiments.utils import create_directory
from src.conformalise import ConformaliseBase
from src.distributions import StudentTMarginals
from src.gps import ExactGP, svGP
from src.projected_langevin_sampling import ProjectedLangevinSampling
from src.temper import TemperBase
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
    elif isinstance(prediction, torch.distributions.Poisson):
        return prediction.rate.sub(y).abs().mean().item()
    elif isinstance(prediction, StudentTMarginals):
        return prediction.loc.sub(y).abs().mean().item()
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
    elif isinstance(prediction, torch.distributions.Poisson):
        return prediction.rate.sub(y).pow(2).mean().item()
    elif isinstance(prediction, StudentTMarginals):
        return prediction.loc.sub(y).pow(2).mean().item()
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
    elif isinstance(prediction, torch.distributions.Poisson):
        return torch.nn.functional.poisson_nll_loss(
            input=prediction.rate,
            target=y.double(),
            reduction="mean",
        ).item()
    elif isinstance(prediction, StudentTMarginals):
        return prediction.negative_log_likelihood(y)
    else:
        raise ValueError(f"Prediction type {type(prediction)} not supported")


def calculate_coverage(
    prediction: torch.distributions.Distribution,
    y: torch.Tensor,
    coverage: float = 0.95,
) -> float:
    if isinstance(prediction, gpytorch.distributions.MultivariateNormal):
        confidence_interval_scale = scipy.special.ndtri((coverage + 1) / 2)
        lower_bound = prediction.mean - confidence_interval_scale * torch.sqrt(
            prediction.variance
        )
        upper_bound = prediction.mean + confidence_interval_scale * torch.sqrt(
            prediction.variance
        )
        return ((lower_bound <= y) & (y <= upper_bound)).float().mean().item()
    else:
        raise ValueError(f"Prediction type {type(prediction)} not supported")


def calculate_average_interval_width(
    model: ConformaliseBase,
    x: torch.Tensor,
    coverage: float,
    y_std: float,
) -> float:
    return model.calculate_average_interval_width(
        x=x,
        coverage=coverage,
    ).item()


def calculate_metrics(
    model: Union[
        ExactGP, svGP, ProjectedLangevinSampling, TemperBase, ConformaliseBase
    ],
    experiment_data: ExperimentData,
    model_name: str,
    dataset_name: str,
    results_path: str,
    plots_path: str,
    particles: torch.Tensor | None = None,
):
    create_directory(os.path.join(results_path, model_name))
    for data in [
        experiment_data.train,
        experiment_data.test,
    ]:
        experiment_data.train.x.cpu()
        experiment_data.train.y.cpu()
        experiment_data.test.x.cpu()
        experiment_data.test.y.cpu()

        set_seed(0)
        if isinstance(model, svGP) or isinstance(model, ExactGP):
            prediction = model.likelihood(model(data.x))
        elif isinstance(model, TemperBase) or isinstance(model, ConformaliseBase):
            prediction = model(data.x)
        elif isinstance(model, ProjectedLangevinSampling):
            prediction = model(x=data.x, particles=particles)
        else:
            raise ValueError(f"Model type {type(model)} not supported")
        mae = calculate_mae(
            y=data.y,
            prediction=prediction,
        )
        pd.DataFrame([[mae]], columns=[model_name], index=[dataset_name]).to_csv(
            os.path.join(results_path, model_name, f"mae_{data.name}.csv"),
            index_label="dataset",
        )
        mse = calculate_mse(
            y=data.y,
            prediction=prediction,
        )
        pd.DataFrame([[mse]], columns=[model_name], index=[dataset_name]).to_csv(
            os.path.join(results_path, model_name, f"mse_{data.name}.csv"),
            index_label="dataset",
        )
        if isinstance(prediction, gpytorch.distributions.MultivariateNormal):
            coverage = calculate_coverage(
                prediction=prediction,
                y=data.y,
                coverage=0.95,
            )
            pd.DataFrame(
                [[coverage]], columns=[model_name], index=[dataset_name]
            ).to_csv(
                os.path.join(results_path, model_name, f"coverage_{data.name}.csv"),
                index_label="dataset",
            )
        if (
            isinstance(prediction, gpytorch.distributions.MultivariateNormal)
            or isinstance(prediction, torch.distributions.Bernoulli)
            or isinstance(prediction, torch.distributions.Poisson)
        ):
            nll = calculate_nll(
                prediction=prediction,
                y=data.y,
            )
            pd.DataFrame([[nll]], columns=[model_name], index=[dataset_name]).to_csv(
                os.path.join(results_path, model_name, f"nll_{data.name}.csv"),
                index_label="dataset",
            )
        if isinstance(prediction, torch.distributions.Bernoulli):
            acc = sklearn.metrics.accuracy_score(
                y_true=data.y.cpu().detach().numpy(),
                y_pred=prediction.probs.round().cpu().detach().numpy(),
            )
            pd.DataFrame(
                [[acc]],
                columns=[model_name],
                index=[dataset_name],
            ).to_csv(
                os.path.join(results_path, model_name, f"acc_{data.name}.csv"),
                index_label="dataset",
            )
            auc = sklearn.metrics.roc_auc_score(
                y_true=data.y.cpu().detach().numpy(),
                y_score=prediction.probs.cpu().detach().numpy(),
            )
            pd.DataFrame(
                [[auc]],
                columns=[model_name],
                index=[dataset_name],
            ).to_csv(
                os.path.join(results_path, model_name, f"auc_{data.name}.csv"),
                index_label="dataset",
            )
            f1 = sklearn.metrics.f1_score(
                y_true=data.y.cpu().detach().numpy(),
                y_pred=prediction.probs.round().cpu().detach().numpy(),
            )
            pd.DataFrame(
                [[f1]],
                columns=[model_name],
                index=[dataset_name],
            ).to_csv(
                os.path.join(results_path, model_name, f"f1_{data.name}.csv"),
                index_label="dataset",
            )

        if isinstance(model, ConformaliseBase):
            average_interval_width = calculate_average_interval_width(
                model=model,
                x=data.x,
                coverage=0.95,
                y_std=experiment_data.y_std,
            )
            pd.DataFrame(
                [[average_interval_width]],
                columns=[model_name],
                index=[dataset_name],
            ).to_csv(
                os.path.join(
                    results_path,
                    model_name,
                    f"average_interval_width_{data.name}.csv",
                ),
                index_label="dataset",
            )

        create_directory(os.path.join(plots_path, model_name))
        plot_true_versus_predicted(
            y_true=data.y,
            y_pred=prediction,
            title=f"True versus Predicted ({mae=:.2f},{mse=:.2f},{nll=:.2f}) ({dataset_name},{model_name},{data.name} data)",
            save_path=os.path.join(
                plots_path, model_name, f"true_versus_predicted_{data.name}.png"
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
