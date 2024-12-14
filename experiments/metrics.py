import os
from typing import List, Union

import gpytorch
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from experiments.data import ExperimentData
from experiments.plotters import plot_true_versus_predicted
from experiments.utils import create_directory
from src.conformalise import ConformaliseBase
from src.conformalise.base import ConformalPrediction
from src.distributions import StudentTMarginals
from src.gaussian_process import ExactGP, svGP
from src.projected_langevin_sampling import PLS
from src.temper import TemperBase
from src.utils import set_seed


def calculate_mae(
    prediction: torch.distributions.Distribution
    | ConformalPrediction
    | StudentTMarginals,
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
    elif isinstance(prediction, torch.distributions.studentT.StudentT):
        return prediction.loc.sub(y).abs().mean().item()
    elif isinstance(prediction, ConformalPrediction):
        return prediction.mean.sub(y).abs().mean().item()
    else:
        raise ValueError(f"Prediction type {type(prediction)} not supported")


def calculate_mse(
    prediction: torch.distributions.Distribution
    | ConformalPrediction
    | StudentTMarginals,
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
    elif isinstance(prediction, torch.distributions.studentT.StudentT):
        return prediction.loc.sub(y).pow(2).mean().item()
    elif isinstance(prediction, ConformalPrediction):
        return prediction.mean.sub(y).pow(2).mean().item()
    else:
        raise ValueError(f"Prediction type {type(prediction)} not supported")


def calculate_nll(
    prediction: torch.distributions.Distribution
    | ConformalPrediction
    | StudentTMarginals,
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
        return prediction.negative_log_likelihood(y).item()
    elif isinstance(prediction, torch.distributions.studentT.StudentT):
        return prediction.log_prob(y).mean().item()
    elif isinstance(prediction, ConformalPrediction):
        assert (
            prediction.coverage == 2 / 3
        ), f"NLL calculation needs 2/3 coverage, got {prediction.coverage=}"
        # average the NLLs for each point by taking half the width of the coverage interval
        # as the standard deviation
        lower, upper = prediction.lower, prediction.upper
        std = (upper - lower) / 2
        return np.mean(
            [
                -scipy.stats.norm.logpdf(y_i, loc=mean_i, scale=std_i)
                for y_i, mean_i, std_i in zip(
                    y.cpu().detach().numpy(),
                    prediction.mean.cpu().detach().numpy(),
                    std.cpu().detach().numpy(),
                )
            ]
        ).item()
    else:
        raise ValueError(f"Prediction type {type(prediction)} not supported")


def calculate_coverage(
    prediction: ConformalPrediction,
    y: torch.Tensor,
) -> float:
    return ((prediction.lower <= y) & (y <= prediction.upper)).float().mean().item()


def calculate_average_interval_width(
    model: ConformaliseBase,
    x: torch.Tensor,
    coverage: float,
) -> float:
    return model.calculate_average_interval_width(
        x=x,
        coverage=coverage,
    )


def calculate_median_interval_width(
    model: ConformaliseBase,
    x: torch.Tensor,
    coverage: float,
) -> float:
    lower, upper = model.predict_coverage(x=x, coverage=coverage)
    return torch.median(upper - lower).item()


def calculate_metrics(
    model: Union[
        ExactGP, svGP, PLS, TemperBase, ConformaliseBase
    ],
    experiment_data: ExperimentData,
    model_name: str,
    dataset_name: str,
    results_path: str,
    plots_path: str,
    coverage: float,
    particles: torch.Tensor | None = None,
):
    assert experiment_data.train is not None
    assert experiment_data.test is not None

    create_directory(os.path.join(results_path, model_name))
    for data in [
        experiment_data.train,
        experiment_data.test,
    ]:
        assert data is not None
        assert data.x is not None
        assert data.y is not None

        experiment_data.train.x.cpu()
        experiment_data.train.y.cpu()
        experiment_data.test.x.cpu()
        experiment_data.test.y.cpu()

        set_seed(0)
        if isinstance(model, svGP) or isinstance(model, ExactGP):
            prediction = model.likelihood(model(data.x))
        elif isinstance(model, ConformaliseBase):
            prediction = model(x=data.x, coverage=coverage)
        elif isinstance(model, PLS) and particles is not None:
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
        if isinstance(prediction, ConformalPrediction):
            prediction_coverage = calculate_coverage(
                prediction=prediction,
                y=data.y,
            )
            pd.DataFrame(
                [[prediction_coverage]], columns=[model_name], index=[dataset_name]
            ).to_csv(
                os.path.join(results_path, model_name, f"coverage_{data.name}.csv"),
                index_label="dataset",
            )
        if (
            isinstance(prediction, gpytorch.distributions.MultivariateNormal)
            or isinstance(prediction, torch.distributions.Bernoulli)
            or isinstance(prediction, torch.distributions.Poisson)
            or isinstance(prediction, ConformalPrediction)
            or isinstance(prediction, StudentTMarginals)
            or isinstance(prediction, torch.distributions.studentT.StudentT)
        ):
            if isinstance(model, ConformaliseBase):
                nll = calculate_nll(
                    prediction=model(x=data.x, coverage=2 / 3),
                    y=data.y,
                )
            else:
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
            median_interval_width = calculate_median_interval_width(
                model=model,
                x=data.x,
                coverage=coverage,
            )
            pd.DataFrame(
                [[median_interval_width]],
                columns=[model_name],
                index=[dataset_name],
            ).to_csv(
                os.path.join(
                    results_path,
                    model_name,
                    f"median_interval_width_{data.name}.csv",
                ),
                index_label="dataset",
            )

            average_interval_width = calculate_average_interval_width(
                model=model,
                x=data.x,
                coverage=coverage,
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
        if not isinstance(prediction, torch.distributions.studentT.StudentT):
            plot_true_versus_predicted(
                y_true=data.y,
                y_pred=prediction,
                title=f"True versus Predicted ({mae=:.2f},{mse=:.2f},{nll=:.2f}) ({dataset_name},{model_name},{data.name} data)",
                save_path=os.path.join(
                    plots_path, model_name, f"true_versus_predicted_{data.name}.png"
                ),
                coverage=coverage,
                error_bar=isinstance(prediction, ConformalPrediction)
                or isinstance(prediction, gpytorch.distributions.MultivariateNormal),
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
