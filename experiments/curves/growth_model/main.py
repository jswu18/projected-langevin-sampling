import argparse
import math
import os
from typing import Any, Dict, Tuple

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
from experiments.loaders import load_pls
from experiments.plotters import plot_1d_experiment_data
from experiments.preprocess import set_up_experiment
from experiments.runners import (
    animate_pls_1d_particles_runner,
    exact_gp_runner,
    inducing_points_runner,
    plot_pls_1d_particles_runner,
    train_pls_runner,
)
from experiments.utils import create_directory
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.kernels.projected_langevin_sampling import PLSKernel
from src.projected_langevin_sampling import ProjectedLangevinSampling
from src.projected_langevin_sampling.basis import InducingPointBasis, OrthonormalBasis
from src.projected_langevin_sampling.costs import LogisticGrowthCost
from src.projected_langevin_sampling.link_functions import LogisticGrowthLinkFunction
from src.utils import set_seed

parser = argparse.ArgumentParser(
    description="Main script for toy Logistic Growth regression experiments."
)
parser.add_argument("--config_path", type=str)
parser.add_argument(
    "--include_gif",
    type=bool,
    default=False,
    help="Indicate whether to include GIFs in the output.",
)


def get_experiment_data(
    seed: int,
    number_of_data_points: int,
    observation_noise: float,
    end_time: float,
    train_data_percentage,
    validation_data_percentage,
    start_time: float = 0,
    initial_population_size: float = 0.1,
) -> ExperimentData:
    generator = torch.Generator().manual_seed(seed)
    time = torch.linspace(start_time, end_time, number_of_data_points + 1)
    step_size = time[1] - time[0]

    noise_vector = torch.normal(
        mean=0.0,
        std=observation_noise,
        size=(number_of_data_points,),
        generator=generator,
    )
    growth_rate = torch.zeros(number_of_data_points)
    population_size = torch.zeros(number_of_data_points + 1)
    population_size[0] = initial_population_size
    for n in range(1, number_of_data_points + 1):
        population_size[n] = (
            population_size[n - 1]
            + step_size * population_size[n - 1] * (1 - population_size[n - 1])
            + step_size * noise_vector[n - 1]
        )
        growth_rate[n - 1] = (population_size[n] - population_size[n - 1]) / step_size

    experiment_data = set_up_experiment(
        name="logistic-curve",
        problem_type=ProblemType.LOGISTIC_GROWTH,
        seed=seed,
        x=time[1:].reshape(-1, 1),
        y=growth_rate,
        train_data_percentage=train_data_percentage,
        validation_data_percentage=validation_data_percentage,
    )
    experiment_data.full.y_untransformed = population_size[1:]
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
    include_gif: bool,
) -> None:
    experiment_data = get_experiment_data(
        seed=data_config["seed"],
        number_of_data_points=data_config["number_of_data_points"],
        observation_noise=data_config["observation_noise"],
        end_time=data_config["end_time"],
        train_data_percentage=data_config["train_data_percentage"],
        validation_data_percentage=data_config["validation_data_percentage"],
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
    subsample_gp_models = exact_gp_runner(
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
        early_stopper_patience=kernel_config["early_stopper_patience"],
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
    inducing_points = inducing_points_runner(
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
    cost = LogisticGrowthCost(
        observation_noise=float(likelihood.noise),
        y_train=experiment_data.train.y,
        link_function=LogisticGrowthLinkFunction(),
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
        particles = torch.divide(1, (1 + torch.exp(-particles)))
        plot_pls_1d_particles_runner(
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
            particles, best_lr, number_of_epochs = train_pls_runner(
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
        plot_pls_1d_particles_runner(
            pls=pls,
            particles=particles,
            particle_name=f"{pls_name}-learned",
            experiment_data=experiment_data,
            plot_particles_path=plot_curve_path,
            plot_title=plot_title,
        )
        if include_gif:
            animate_pls_1d_particles_runner(
                pls=pls,
                number_of_particles=pls_config["number_of_particles"],
                particle_name=pls_name,
                experiment_data=experiment_data,
                seed=pls_config["seed"],
                best_lr=best_lr,
                number_of_epochs=number_of_epochs,
                plot_title=plot_title,
                animate_1d_path=None,
                animate_1d_untransformed_path=plot_curve_path,
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
            include_gif=args.include_gif,
        )
        break
