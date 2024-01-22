import argparse
import math
import os
from typing import Any, Dict

import gpytorch
import matplotlib.pyplot as plt
import torch
import yaml

from experiments.constructors import (
    construct_average_ard_kernel,
    construct_average_gaussian_likelihood,
)
from experiments.curves.curves import CURVE_FUNCTIONS, Curve
from experiments.data import Data, ExperimentData, ProblemType
from experiments.loaders import load_pls, load_svgp
from experiments.metrics import calculate_metrics, concatenate_metrics
from experiments.plotters import plot_1d_experiment_data
from experiments.preprocess import split_regression_data_intervals
from experiments.runners import (
    learn_subsample_gps,
    select_inducing_points,
    train_pls,
    train_svgp,
)
from experiments.utils import create_directory
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.kernels.projected_langevin_sampling import PLSKernel
from src.projected_langevin_sampling import (
    PLSPoissonRegressionIPB,
    PLSPoissonRegressionONB,
)
from src.utils import set_seed

parser = argparse.ArgumentParser(
    description="Main script for toy regression curves experiments."
)
parser.add_argument("--config_path", type=str)


def get_experiment_data(
    curve_function: Curve,
    number_of_data_points: int,
    seed: int,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
) -> ExperimentData:
    x = torch.linspace(-3, 3, number_of_data_points).reshape(-1, 1)
    y_curve = 2 * curve_function.calculate_curve(
        x=x,
    ).reshape(-1)
    generator = torch.Generator().manual_seed(seed)
    y = torch.poisson(torch.pow(y_curve, 2), generator=generator).reshape(-1)
    y_untransformed = torch.pow(y_curve, 2)
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
        problem_type=ProblemType.REGRESSION,
        full=Data(
            x=x,
            y=y.type(torch.int),
            y_untransformed=y_untransformed,
            name="full",
        ),
        train=Data(
            x=x_train,
            y=y_train.type(torch.int),
            y_untransformed=y_train_untransformed,
            name="train",
        ),
        test=Data(
            x=x_test,
            y=y_test.type(torch.int),
            y_untransformed=y_test_untransformed,
            name="test",
        ),
    )
    return experiment_data


def plot_experiment_data(
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
    ax.set_title(title)
    fig.tight_layout()
    create_directory(
        f"experiments/curves/poisson_regression/outputs/plots/{curve_name}"
    )
    plt.savefig(
        f"experiments/curves/poisson_regression/outputs/plots/{curve_name}/experiment-data.png"
    )
    plt.close()


def main(
    curve_function: Curve,
    data_config: Dict[str, Any],
    kernel_config: Dict[str, Any],
    inducing_points_config: Dict[str, Any],
    pls_config: Dict[str, Any],
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
        experiment_data=experiment_data,
        title=f"{curve_function.__name__} data",
        curve_name=type(curve_function).__name__.lower(),
    )
    plot_curve_path = f"experiments/curves/poisson_regression/outputs/plots/{type(curve_function).__name__.lower()}"
    results_curve_path = f"experiments/curves/poisson_regression/outputs/results/{type(curve_function).__name__.lower()}"
    models_path = f"experiments/curves/poisson_regression/outputs/models/{type(curve_function).__name__.lower()}"
    data_path = f"experiments/curves/poisson_regression/outputs/data/{type(curve_function).__name__.lower()}"
    subsample_gp_model_path = os.path.join(models_path, "subsample_gp")
    subsample_gp_data_path = os.path.join(data_path, "subsample_gp")

    subsample_gp_models = learn_subsample_gps(
        experiment_data=experiment_data,
        kernel=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=experiment_data.train.x.shape[1])
        ),
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        subsample_size=kernel_config["subsample_size"],
        seed=kernel_config["seed"],
        number_of_epochs=kernel_config["number_of_epochs"],
        learning_rate=kernel_config["learning_rate"],
        number_of_iterations=kernel_config["number_of_iterations"],
        model_path=subsample_gp_model_path,
        data_path=subsample_gp_data_path,
        plot_1d_subsample_path=None,
        plot_loss_path=plot_curve_path,
    )
    average_ard_kernel = construct_average_ard_kernel(
        kernels=[model.kernel for model in subsample_gp_models]
    )
    likelihood = construct_average_gaussian_likelihood(
        likelihoods=[model.likelihood for model in subsample_gp_models]
    )
    inducing_points = select_inducing_points(
        seed=inducing_points_config["seed"],
        inducing_point_selector=ConditionalVarianceInducingPointSelector(),
        data=experiment_data.train,
        number_induce_points=int(
            inducing_points_config["inducing_points_factor"]
            * math.pow(
                experiment_data.train.x.shape[0],
                1 / inducing_points_config["inducing_points_power"],
            )
        ),
        kernel=average_ard_kernel,
    )
    pls_kernel = PLSKernel(
        base_kernel=average_ard_kernel,
        approximation_samples=inducing_points.x,
    )
    pls_dict = {
        "pls-onb": PLSPoissonRegressionONB(
            kernel=pls_kernel,
            x_induce=inducing_points.x,
            y_induce=inducing_points.y,
            x_train=experiment_data.train.x,
            y_train=experiment_data.train.y,
            jitter=pls_config["jitter"],
        ),
        "pls-ipb": PLSPoissonRegressionIPB(
            kernel=pls_kernel,
            x_induce=inducing_points.x,
            y_induce=inducing_points.y,
            x_train=experiment_data.train.x,
            y_train=experiment_data.train.y,
            jitter=pls_config["jitter"],
        ),
    }
    for pls_name, pls in pls_dict.items():
        pls_path = os.path.join(models_path, f"{pls_name}.pth")
        # if os.path.exists(pls_path):
        #     pls, particles = load_pls(
        #         pls=pls,
        #         model_path=pls_path,
        #     )
        # else:
        particles = train_pls(
            pls=pls,
            number_of_particles=pls_config["number_of_particles"],
            particle_name=pls_name,
            experiment_data=experiment_data,
            inducing_points=inducing_points,
            simulation_duration=pls_config["simulation_duration"],
            step_size_upper=pls_config["step_size_upper"],
            number_of_step_searches=pls_config["number_of_step_searches"],
            maximum_number_of_steps=pls_config["maximum_number_of_steps"],
            minimum_change_in_energy_potential=pls_config[
                "minimum_change_in_energy_potential"
            ],
            seed=pls_config["seed"],
            observation_noise_upper=pls_config["observation_noise_upper"],
            observation_noise_lower=pls_config["observation_noise_lower"],
            number_of_observation_noise_searches=pls_config[
                "number_of_observation_noise_searches"
            ],
            plot_title=curve_function.__name__,
            plot_particles_path=plot_curve_path,
            # animate_1d_path=plot_curve_path,
            # animate_1d_untransformed_path=plot_curve_path,
            plot_update_magnitude_path=plot_curve_path,
            christmas_colours=pls_config["christmas_colours"]
            if "christmas_colours" in pls_config
            else False,
            metric_to_minimise=pls_config["metric_to_minimise"],
            initial_particles_noise_only=pls_config["initial_particles_noise_only"],
            early_stopper_patience=pls_config["early_stopper_patience"],
        )
        torch.save(
            {
                "particles": particles,
                "observation_noise": pls.observation_noise,
            },
            pls_path,
        )
        # set_seed(pls_config["seed"])
        # calculate_metrics(
        #     model=pls,
        #     particles=particles,
        #     model_name=pls_name,
        #     dataset_name=type(curve_function).__name__,
        #     experiment_data=experiment_data,
        #     results_path=results_curve_path,
        #     plots_path=plot_curve_path,
        # )

    # for kernel_name, kernel in zip(["k", "r"], [average_ard_kernel, pls_kernel]):
    #     model_name = f"svgp-{kernel_name}"
    #     svgp_model_path = os.path.join(models_path, f"{model_name}.pth")
    #     if os.path.exists(svgp_model_path):
    #         svgp, _ = load_svgp(
    #             model_path=svgp_model_path,
    #             x_induce=inducing_points.x,
    #             mean=gpytorch.means.ConstantMean(),
    #             kernel=deepcopy(kernel),
    #             likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    #             learn_inducing_locations=False,
    #         )
    #     else:
    #         svgp, losses = train_svgp(
    #             model_name=model_name,
    #             experiment_data=experiment_data,
    #             inducing_points=inducing_points,
    #             mean=gpytorch.means.ConstantMean(),
    #             kernel=deepcopy(kernel),
    #             likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    #             seed=svgp_config["seed"],
    #             number_of_epochs=svgp_config["number_of_epochs"],
    #             batch_size=svgp_config["batch_size"],
    #             learning_rate_upper=svgp_config["learning_rate_upper"],
    #             learning_rate_lower=svgp_config["learning_rate_lower"],
    #             number_of_learning_rate_searches=svgp_config[
    #                 "number_of_learning_rate_searches"
    #             ],
    #             is_fixed=True,
    #             observation_noise=float(likelihood.noise),
    #             models_path=os.path.join(
    #                 models_path, f"{model_name}-kernel-iterations"
    #             ),
    #             plot_title=curve_function.__name__,
    #             plot_1d_path=plot_curve_path,
    #             # animate_1d_path=plot_curve_path,
    #             plot_loss_path=plot_curve_path,
    #             christmas_colours=svgp_config["christmas_colours"]
    #             if "christmas_colours" in pls_config
    #             else False,
    #         )
    #         torch.save(
    #             {
    #                 "model": svgp.state_dict(),
    #                 "losses": losses,
    #             },
    #             os.path.join(models_path, f"{model_name}.pth"),
    #         )
    #     set_seed(svgp_config["seed"])
    #     calculate_metrics(
    #         model=svgp,
    #         model_name=model_name,
    #         dataset_name=type(curve_function).__name__,
    #         experiment_data=experiment_data,
    #         results_path=results_curve_path,
    #         plots_path=plot_curve_path,
    #     )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    for curve_function_ in CURVE_FUNCTIONS:
        main(
            curve_function=curve_function_,
            data_config=loaded_config["data"],
            kernel_config=loaded_config["kernel"],
            inducing_points_config=loaded_config["inducing_points"],
            pls_config=loaded_config["pls"],
            svgp_config=loaded_config["svgp"],
        )
    # concatenate_metrics(
    #     results_path="experiments/curves/poisson_regression/outputs/results",
    #     data_types=["train", "test"],
    #     model_names=["pls-onb", "pls-ipb", "svgp-k", "svgp-r"],
    #     datasets=[
    #         type(curve_function_).__name__.lower()
    #         for curve_function_ in CURVE_FUNCTIONS
    #     ],
    #     metrics=["mae", "mse", "nll"],
    # )
