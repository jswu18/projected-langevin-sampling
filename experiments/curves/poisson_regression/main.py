import argparse
import math
import os
from typing import Any, Dict

import gpytorch
import matplotlib.pyplot as plt
import torch
import yaml

from experiments.constructors import construct_average_ard_kernel
from experiments.curves.curves import CURVE_FUNCTIONS, Curve
from experiments.data import Data, ExperimentData, ProblemType
from experiments.loaders import load_pls
from experiments.plotters import plot_1d_experiment_data
from experiments.preprocess import split_regression_data_intervals
from experiments.runners import (
    animate_pls_1d_particles,
    learn_subsample_gps,
    plot_pls_1d_particles,
    select_inducing_points,
    train_pls,
)
from experiments.utils import create_directory
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.kernels.projected_langevin_sampling import PLSKernel
from src.projected_langevin_sampling import ProjectedLangevinSampling
from src.projected_langevin_sampling.basis import InducingPointBasis, OrthonormalBasis
from src.projected_langevin_sampling.costs import PoissonCost
from src.projected_langevin_sampling.link_functions import SquareLinkFunction
from src.utils import set_seed

parser = argparse.ArgumentParser(
    description="Main script for toy Poisson regression experiments."
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
    link_function = SquareLinkFunction()
    y = torch.poisson(link_function.transform(y_curve), generator=generator).reshape(-1)
    y_untransformed = link_function.transform(y_curve)
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
        problem_type=ProblemType.POISSON_REGRESSION,
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
    plot_curve_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    ax.set_title(title)
    fig.tight_layout()
    create_directory(plot_curve_path)
    plt.savefig(os.path.join(plot_curve_path, "experiment-data.png"))
    plt.close()


def main(
    curve_function: Curve,
    data_config: Dict[str, Any],
    kernel_config: Dict[str, Any],
    inducing_points_config: Dict[str, Any],
    pls_config: Dict[str, Any],
    outputs_path: str,
) -> None:
    experiment_data = get_experiment_data(
        curve_function=curve_function,
        number_of_data_points=data_config["number_of_data_points"],
        seed=data_config["seed"],
        number_of_test_intervals=data_config["number_of_test_intervals"],
        total_number_of_intervals=data_config["total_number_of_intervals"],
    )
    data_path = os.path.join(
        outputs_path, "data", type(curve_function).__name__.lower()
    )
    plot_curve_path = os.path.join(
        outputs_path, "plots", type(curve_function).__name__.lower()
    )
    models_path = os.path.join(
        outputs_path, "models", type(curve_function).__name__.lower()
    )
    plot_experiment_data(
        experiment_data=experiment_data,
        title=f"{curve_function.__name__} data",
        plot_curve_path=plot_curve_path,
    )
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
    onb_basis = OrthonormalBasis(
        kernel=pls_kernel,
        x_induce=inducing_points.x,
        x_train=experiment_data.train.x,
    )
    ipb_basis = InducingPointBasis(
        kernel=pls_kernel,
        x_induce=inducing_points.x,
        y_induce=inducing_points.y,
        x_train=experiment_data.train.x,
    )
    cost = PoissonCost(
        y_train=experiment_data.train.y,
        link_function=SquareLinkFunction(),
    )
    pls_dict = {
        "pls-onb": ProjectedLangevinSampling(
            basis=onb_basis,
            cost=cost,
        ),
        "pls-ipb": ProjectedLangevinSampling(
            basis=ipb_basis,
            cost=cost,
        ),
    }
    plot_title = "PLS for Poisson Regression"
    for pls_name, pls in pls_dict.items():
        pls_path = os.path.join(models_path, f"{pls_name}.pth")
        set_seed(pls_config["seed"])
        particles = pls.initialise_particles(
            number_of_particles=pls_config["number_of_particles"],
            seed=pls_config["seed"],
            noise_only=pls_config["initial_particles_noise_only"],
        )
        plot_pls_1d_particles(
            pls=pls,
            particles=particles,
            particle_name=f"{pls_name}-initial",
            experiment_data=experiment_data,
            plot_particles_path=plot_curve_path,
            plot_title=plot_title,
        )
        if os.path.exists(pls_path):
            pls, particles, best_lr, number_of_epochs = load_pls(
                pls=pls,
                model_path=pls_path,
            )
        else:
            particles, best_lr, number_of_epochs = train_pls(
                pls=pls,
                particles=particles,
                particle_name=pls_name,
                experiment_data=experiment_data,
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
                plot_title=plot_title,
                plot_energy_potential_path=plot_curve_path,
                metric_to_optimise=pls_config["metric_to_optimise"],
                early_stopper_patience=pls_config["early_stopper_patience"],
            )
            torch.save(
                {
                    "particles": particles,
                    "observation_noise": pls.observation_noise,
                    "best_lr": best_lr,
                    "number_of_epochs": number_of_epochs,
                },
                pls_path,
            )
        plot_pls_1d_particles(
            pls=pls,
            particles=particles,
            particle_name=f"{pls_name}-learned",
            experiment_data=experiment_data,
            plot_particles_path=plot_curve_path,
            plot_title=plot_title,
        )
        animate_pls_1d_particles(
            pls=pls,
            number_of_particles=pls_config["number_of_particles"],
            particle_name=pls_name,
            experiment_data=experiment_data,
            seed=pls_config["seed"],
            best_lr=best_lr,
            number_of_epochs=number_of_epochs,
            animate_1d_path=plot_curve_path,
            plot_title=plot_title,
            animate_1d_untransformed_path=None,
            christmas_colours=pls_config["christmas_colours"]
            if "christmas_colours" in pls_config
            else False,
            initial_particles_noise_only=pls_config["initial_particles_noise_only"],
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    outputs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
    for curve_function_ in CURVE_FUNCTIONS:
        main(
            curve_function=curve_function_,
            data_config=loaded_config["data"],
            kernel_config=loaded_config["kernel"],
            inducing_points_config=loaded_config["inducing_points"],
            pls_config=loaded_config["pls"],
            outputs_path=outputs_path,
        )
