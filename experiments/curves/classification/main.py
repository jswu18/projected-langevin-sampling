import argparse
import math
import os
from typing import Any, Dict

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from experiments.constructors import (
    construct_average_ard_kernel,
    construct_average_dirichlet_likelihood,
    construct_average_gaussian_likelihood,
)
from experiments.curves.curves import CURVE_FUNCTIONS, Curve
from experiments.data import Data, ExperimentData
from experiments.metrics import calculate_metrics, concatenate_metrics
from experiments.plotters import plot_1d_experiment_data
from experiments.preprocess import split_regression_data_intervals
from experiments.runners import (
    learn_subsample_gps,
    pwgf_observation_noise_search,
    select_induce_data,
    train_projected_wasserstein_gradient_flow,
    train_svgp,
)
from experiments.utils import create_directory
from src.gradient_flows import (
    GradientFlowClassificationNONB,
    GradientFlowClassificationONB,
)
from src.gradient_flows.base.transforms.classification import (
    GradientFlowClassificationBase,
)
from src.induce_data_selectors import ConditionalVarianceInduceDataSelector
from src.kernels import GradientFlowKernel

parser = argparse.ArgumentParser(
    description="Main script for toy binary classification experiments."
)
parser.add_argument("--config_path", type=str)


def get_experiment_data(
    curve_function: Curve,
    number_of_data_points: int,
    seed: int,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
) -> ExperimentData:
    x = torch.linspace(-2, 2, number_of_data_points).reshape(-1, 1)
    y_curve = curve_function.calculate_curve(
        x=x,
    ).reshape(-1)
    # y = curve_function.classification(
    #     y_curve=y_curve,
    # )
    y = curve_function.classification(
        y_curve=y_curve,
        seed=seed,
    )
    y_untransformed = GradientFlowClassificationBase.transform(y=y_curve)
    (
        x_train,
        y_train,
        y_train_untransformed,
        x_test,
        y_test,
        y_test_untransformed,
    ) = split_regression_data_intervals(
        split_seed=curve_function.seed,
        x=x,
        y=y,
        y_untransformed=y_untransformed,
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
    )
    experiment_data = ExperimentData(
        name=type(curve_function).__name__.lower(),
        full=Data(
            x=x.double(),
            y=y.type(torch.int),
            y_untransformed=y_untransformed,
            name="full",
        ),
        train=Data(
            x=x_train.double(),
            y=y_train.type(torch.int),
            y_untransformed=y_train_untransformed,
            name="train",
        ),
        test=Data(
            x=x_test.double(),
            y=y_test.type(torch.int),
            y_untransformed=y_test_untransformed,
            name="test",
        ),
    )
    return experiment_data


def plot_experiment_data(
    curve_function: Curve,
    experiment_data: ExperimentData,
    title: str,
    curve_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    create_directory(f"experiments/curves/classification/outputs/plots/{curve_name}")
    plt.savefig(
        f"experiments/curves/classification/outputs/plots/{curve_name}/experiment-data.png"
    )
    plt.close()


def main(
    curve_function: Curve,
    data_config: Dict[str, Any],
    kernel_config: Dict[str, Any],
    induce_data_config: Dict[str, Any],
    pwgf_config: Dict[str, Any],
    svgp_config: Dict[str, Any],
) -> None:
    experiment_data = get_experiment_data(
        curve_function=curve_function,
        number_of_data_points=data_config["number_of_data_points"],
        seed=data_config["seed"],
        number_of_test_intervals=data_config["number_of_test_intervals"],
        total_number_of_intervals=data_config["total_number_of_intervals"],
    )
    plot_experiment_data(
        curve_function=curve_function,
        experiment_data=experiment_data,
        title=f"{curve_function.__name__} data",
        curve_name=type(curve_function).__name__.lower(),
    )
    plot_curve_path = f"experiments/curves/classification/outputs/plots/{type(curve_function).__name__.lower()}"
    results_curve_path = f"experiments/curves/classification/outputs/results/{type(curve_function).__name__.lower()}"
    models_path = f"experiments/curves/classification/outputs/models/{type(curve_function).__name__.lower()}"
    data_path = f"experiments/curves/classification/outputs/data/{type(curve_function).__name__.lower()}"
    subsample_gp_model_path = os.path.join(models_path, "subsample_gp")
    subsample_gp_data_path = os.path.join(data_path, "subsample_gp")
    svgp_iteration_model_path = os.path.join(models_path, "svgp_model_iterations")
    likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
        experiment_data.train.y, learn_additional_noise=True
    )
    y_train_labels = experiment_data.train.y
    experiment_data.train.y = likelihood.transformed_targets
    subsample_gp_models = learn_subsample_gps(
        experiment_data=experiment_data,
        kernel=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=experiment_data.train.x.shape[1],
                batch_shape=torch.Size((likelihood.num_classes,)),
            ),
            batch_shape=torch.Size((likelihood.num_classes,)),
        ),
        likelihood=likelihood,
        subsample_size=kernel_config["subsample_size"],
        seed=kernel_config["seed"],
        number_of_epochs=kernel_config["number_of_epochs"],
        learning_rate=kernel_config["learning_rate"],
        number_of_iterations=kernel_config["number_of_iterations"],
        model_path=subsample_gp_model_path,
        data_path=subsample_gp_data_path,
        plot_1d_subsample_path=None,
        plot_loss_path=plot_curve_path,
        number_of_classes=likelihood.num_classes,
    )
    experiment_data.train.y = y_train_labels
    average_ard_kernel = construct_average_ard_kernel(
        kernels=[model.kernel for model in subsample_gp_models],
    )
    induce_data = select_induce_data(
        seed=induce_data_config["seed"],
        induce_data_selector=ConditionalVarianceInduceDataSelector(),
        data=experiment_data.train,
        number_induce_points=int(
            induce_data_config["induce_data_factor"]
            * math.pow(
                experiment_data.train.x.shape[0],
                1 / induce_data_config["induce_data_power"],
            )
        ),
        kernel=average_ard_kernel,
    )
    gradient_flow_kernel = GradientFlowKernel(
        base_kernel=average_ard_kernel,
        approximation_samples=induce_data.x,
        number_of_classes=likelihood.num_classes,
    )
    pwgf_dict = {
        "pwgf-orthonormal-basis": GradientFlowClassificationONB(
            kernel=gradient_flow_kernel,
            x_induce=induce_data.x,
            y_induce=induce_data.y,
            x_train=experiment_data.train.x,
            y_train=experiment_data.train.y,
            jitter=pwgf_config["jitter"],
        ),
        "pwgf-induce-data-basis": GradientFlowClassificationNONB(
            kernel=gradient_flow_kernel,
            x_induce=induce_data.x,
            y_induce=induce_data.y,
            x_train=experiment_data.train.x,
            y_train=experiment_data.train.y,
            jitter=pwgf_config["jitter"],
        ),
    }
    for pwgf_name, pwgf in pwgf_dict.items():
        particles = train_projected_wasserstein_gradient_flow(
            pwgf=pwgf,
            number_of_particles=pwgf_config["number_of_particles"],
            particle_name=pwgf_name,
            experiment_data=experiment_data,
            induce_data=induce_data,
            simulation_duration=pwgf_config["simulation_duration"],
            seed=pwgf_config["seed"],
            plot_title=f"{type(curve_function).__name__}",
            plot_particles_path=plot_curve_path,
            animate_1d_path=plot_curve_path,
            plot_update_magnitude_path=plot_curve_path,
            christmas_colours=pwgf_config["christmas_colours"]
            if "christmas_colours" in pwgf_config
            else False,
            metric_to_minimise=pwgf_config["metric_to_minimise"],
            initial_particles_noise_only=pwgf_config["initial_particles_noise_only"],
        )
        calculate_metrics(
            model=pwgf,
            particles=particles,
            model_name=pwgf_name,
            dataset_name=type(curve_function).__name__,
            experiment_data=experiment_data,
            results_path=results_curve_path,
            plots_path=plot_curve_path,
        )
    svgp_model, _ = train_svgp(
        model_name="svgp",
        experiment_data=experiment_data,
        induce_data=induce_data,
        mean=gpytorch.means.ConstantMean(),
        kernel=gradient_flow_kernel,
        likelihood=gpytorch.likelihoods.BernoulliLikelihood(),
        seed=svgp_config["seed"],
        number_of_epochs=svgp_config["number_of_epochs"],
        batch_size=svgp_config["batch_size"],
        learning_rate_upper=svgp_config["learning_rate_upper"],
        learning_rate_lower=svgp_config["learning_rate_lower"],
        number_of_learning_rate_searches=svgp_config[
            "number_of_learning_rate_searches"
        ],
        is_fixed=True,
        observation_noise=None,
        models_path=svgp_iteration_model_path,
        plot_title=f"{type(curve_function).__name__}",
        plot_1d_path=plot_curve_path,
        animate_1d_path=plot_curve_path,
        plot_loss_path=plot_curve_path,
        christmas_colours=svgp_config["christmas_colours"]
        if "christmas_colours" in pwgf_config
        else False,
    )
    calculate_metrics(
        model=svgp_model,
        model_name="svgp",
        dataset_name=type(curve_function).__name__,
        experiment_data=experiment_data,
        results_path=results_curve_path,
        plots_path=plot_curve_path,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    for curve_function_ in CURVE_FUNCTIONS:
        main(
            curve_function=curve_function_,
            data_config=loaded_config["data"],
            kernel_config=loaded_config["kernel"],
            induce_data_config=loaded_config["induce_data"],
            pwgf_config=loaded_config["pwgf"],
            svgp_config=loaded_config["svgp"],
        )
