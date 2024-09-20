import argparse
import math
import os
from typing import Any, Dict, List, Tuple

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
from experiments.plotters import (
    plot_1d_experiment_data,
    plot_1d_gp_prediction,
    plot_1d_pls_prediction,
    plot_eigenvalues,
)
from experiments.preprocess import set_up_experiment, split_regression_data_intervals
from experiments.runners import (
    animate_pls_1d_particles_runner,
    exact_gp_runner,
    inducing_points_runner,
    plot_pls_1d_particles_runner,
    train_pls_runner,
)
from experiments.utils import create_directory
from src.conformalise import ConformalisePLS
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.kernels.projected_langevin_sampling import PLSKernel
from src.projected_langevin_sampling import ProjectedLangevinSampling
from src.projected_langevin_sampling.basis import InducingPointBasis, OrthonormalBasis
from src.projected_langevin_sampling.costs import PoissonCost
from src.projected_langevin_sampling.costs.multimodal import MultiModalCost
from src.projected_langevin_sampling.link_functions import (
    IdentityLinkFunction,
    SquareLinkFunction,
)
from src.utils import set_seed

parser = argparse.ArgumentParser(
    description="Main script for toy multi-modal regression experiments."
)
parser.add_argument("--config_path", type=str)
parser.add_argument(
    "--include_gif",
    type=bool,
    default=False,
    help="Indicate whether to include GIFs in the output.",
)


def get_experiment_data(
    curve_function: Curve,
    number_of_data_points: int,
    seed: int,
    bernoulli_probability_true: float,
    bernoulli_shift_true: float,
    sigma_true: float,
    train_data_percentage: float,
    validation_data_percentage: float,
) -> Tuple[ExperimentData, ExperimentData]:
    x = torch.linspace(-3, 3, number_of_data_points).reshape(-1, 1)
    y_curve = 2 * curve_function.calculate_curve(
        x=x,
    ).reshape(-1)
    bernoulli_noise = torch.bernoulli(
        input=torch.ones(y_curve.shape) * bernoulli_probability_true,
        generator=torch.Generator().manual_seed(seed),
    )
    gaussian_noise = torch.normal(
        mean=0.0,
        std=1.0,
        generator=torch.Generator().manual_seed(seed),
        size=y_curve.shape,
    )
    y = y_curve + sigma_true * gaussian_noise

    experiment_data = set_up_experiment(
        name=curve_function.__name__,
        problem_type=ProblemType.MULTIMODAL_REGRESSION,
        seed=seed,
        x=x,
        y=y,
        train_data_percentage=train_data_percentage,
        validation_data_percentage=validation_data_percentage,
        normalise=True,
    )
    experiment_data_bimodal = set_up_experiment(
        name=curve_function.__name__,
        problem_type=ProblemType.MULTIMODAL_REGRESSION,
        seed=seed,
        x=experiment_data.full.x,
        y=experiment_data.full.y + bernoulli_shift_true * bernoulli_noise,
        train_data_percentage=train_data_percentage,
        validation_data_percentage=validation_data_percentage,
        normalise=False,
    )
    return experiment_data_bimodal, experiment_data


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
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    create_directory(plot_curve_path)
    plt.savefig(os.path.join(plot_curve_path, "experiment-data.png"))
    plt.close()


def plot_gp_prediction(
    experiment_data: ExperimentData,
    model: gpytorch.models.ExactGP,
    title: str,
    save_path: str,
) -> None:
    model.eval()
    model.likelihood.eval()
    predicted_distribution = model.likelihood(model(experiment_data.full.x))
    create_directory(os.path.dirname(save_path))
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    _ = plot_1d_gp_prediction(
        fig=fig,
        ax=ax,
        x=experiment_data.full.x,
        mean=predicted_distribution.mean.detach(),
        variance=predicted_distribution.variance.detach(),
        title=title,
        save_path=save_path,
    )


def approximate_bernoulli_parameters(
    train_data: Data,
    subsample_gp_models: List[gpytorch.models.ExactGP],
) -> Tuple[float, float]:
    shifts = []
    bernoulli_noises = []
    for model in subsample_gp_models:
        model.eval()
        model.likelihood.eval()
        predicted_distribution = model.likelihood(model(train_data.x))

        # double the absolute mean error
        shifts.append(
            2
            * (predicted_distribution.mean - train_data.y).abs().mean().detach().item()
        )

        # ratio of points above and below the mean
        bernoulli_noises.append(
            torch.mean((predicted_distribution.mean > train_data.y).float()).item()
        )
    return (
        torch.tensor(shifts).mean().item(),
        torch.tensor(bernoulli_noises).mean().item(),
    )


def main(
    curve_function: Curve,
    data_config: Dict[str, Any],
    kernel_config: Dict[str, Any],
    inducing_points_config: Dict[str, Any],
    pls_config: Dict[str, Any],
    outputs_path: str,
    include_gif: bool,
) -> None:
    experiment_data, experiment_data_gp = get_experiment_data(
        curve_function=curve_function,
        number_of_data_points=data_config["number_of_data_points"],
        seed=data_config["seed"],
        bernoulli_probability_true=data_config["bernoulli_probability_true"],
        bernoulli_shift_true=data_config["bernoulli_shift_true"],
        sigma_true=data_config["sigma_true"],
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
    results_path = os.path.join(
        outputs_path, "results", type(curve_function).__name__.lower()
    )
    plot_experiment_data(
        experiment_data=experiment_data,
        title=f"{curve_function.__name__} data",
        plot_curve_path=plot_curve_path,
    )
    subsample_gp_model_path = os.path.join(models_path, "subsample_gp")
    subsample_gp_data_path = os.path.join(data_path, "subsample_gp")
    plot_experiment_data(
        experiment_data=experiment_data,
        title=f"{curve_function.__name__} data GP",
        plot_curve_path=plot_curve_path,
    )
    subsample_gp_models = exact_gp_runner(
        experiment_data=experiment_data_gp,
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
    plot_gp_prediction(
        experiment_data=experiment_data_gp,
        model=subsample_gp_models[0],
        title=f"{curve_function.__name__} Exact GP",
        save_path=os.path.join(plot_curve_path, "exact_gp.png"),
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
    approximate_shift, approximate_bernoulli_noise = approximate_bernoulli_parameters(
        train_data=experiment_data.train,
        subsample_gp_models=subsample_gp_models,
    )
    # temporary
    y_temp = torch.linspace(-10, 20, 1000).reshape(1, -1)
    cost = MultiModalCost(
        observation_noise=data_config["sigma_true"],
        # observation_noise=1e-1,
        # observation_noise=float(likelihood.noise)/2,
        y_train=y_temp,
        link_function=IdentityLinkFunction(),
        shift=data_config["bernoulli_shift_true"],
        bernoulli_noise=0.5,
    )
    plt.plot(
        y_temp.reshape(-1), cost.calculate_cost(torch.zeros(y_temp.shape)).reshape(-1)
    )
    plt.show()
    #####

    cost = MultiModalCost(
        observation_noise=data_config["sigma_true"],
        # observation_noise=0.01,
        # observation_noise=float(likelihood.noise)/2,
        y_train=experiment_data.train.y,
        link_function=IdentityLinkFunction(),
        shift=data_config["bernoulli_shift_true"],
        bernoulli_noise=data_config["bernoulli_probability_true"],
    )
    plot_title = "PLS for Multi-modal Regression"
    pls = ProjectedLangevinSampling(basis=onb_basis, cost=cost, name="pls-onb")
    set_seed(pls_config["seed"])
    # particles = pls.initialise_particles(
    #     number_of_particles=pls_config["number_of_particles"],
    #     seed=pls_config["seed"],
    #     noise_only=pls_config["initial_particles_noise_only"],
    # )
    particles = torch.normal(
        0,
        0.01,
        size=(onb_basis.approximation_dimension, pls_config["number_of_particles"]),
    )
    bernoulli_samples = torch.bernoulli(
        torch.ones((pls_config["number_of_particles"]))
        * data_config["bernoulli_probability_true"],
    )
    particles += 0.5 * bernoulli_samples[None, :]  # add zero or one to the particles
    particles = (
        math.sqrt(onb_basis.x_induce.shape[0])
        * onb_basis.eigenvectors.T
        @ torch.diag(torch.divide(1, torch.sqrt(onb_basis.eigenvalues)))
        @ particles
    )

    plot_pls_1d_particles_runner(
        pls=pls,
        particles=particles,
        particle_name=f"{pls.name}-initial",
        experiment_data=experiment_data,
        inducing_points=inducing_points,
        plot_particles_path=plot_curve_path,
        plot_title=plot_title,
    )
    plot_eigenvalues(
        basis=onb_basis,
        save_path=os.path.join(plot_curve_path, f"eigenvalues.png"),
        title=f"Eigenvalues",
    )
    # if os.path.exists(pls_path):
    #     pls, particles, best_lr, number_of_epochs = load_pls(
    #         pls=pls,
    #         model_path=pls_path,
    #     )
    # else:
    particles, best_lr, number_of_epochs = train_pls_runner(
        pls=pls,
        particles=particles,
        particle_name=pls.name,
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
    # torch.save(
    #     {
    #         "particles": particles,
    #         "observation_noise": pls.observation_noise,
    #         "best_lr": best_lr,
    #         "number_of_epochs": number_of_epochs,
    #     },
    #     pls_path,
    # )
    plot_pls_1d_particles_runner(
        pls=pls,
        particles=particles,
        particle_name=f"{pls.name}-learned",
        experiment_data=experiment_data,
        inducing_points=inducing_points,
        plot_particles_path=plot_curve_path,
        plot_title=plot_title,
    )
    pls_conformalised = ConformalisePLS(
        x_calibration=experiment_data.validation.x,
        y_calibration=experiment_data.validation.y,
        pls=pls,
        particles=particles,
    )
    plot_pls_1d_particles_runner(
        pls=pls_conformalised,
        particles=particles,
        particle_name=f"{pls.name}-learned-conformalised",
        experiment_data=experiment_data,
        plot_particles_path=plot_curve_path,
        plot_title=f"{plot_title} Conformalised",
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    OUTPUTS_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
    for curve_function_ in CURVE_FUNCTIONS:
        main(
            curve_function=curve_function_,
            data_config=loaded_config["data"],
            kernel_config=loaded_config["kernel"],
            inducing_points_config=loaded_config["inducing_points"],
            pls_config=loaded_config["pls"],
            outputs_path=OUTPUTS_PATH,
            include_gif=args.include_gif,
        )
        break
