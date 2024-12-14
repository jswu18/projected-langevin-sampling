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
from experiments.data import ExperimentData, ProblemType
from experiments.plotters import (
    plot_1d_experiment_data,
    plot_1d_gp_prediction,
    plot_eigenvalues,
)
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
from src.projected_langevin_sampling.basis import OrthonormalBasis
from src.projected_langevin_sampling.costs.multimodal import MultiModalCost
from src.projected_langevin_sampling.link_functions import IdentityLinkFunction
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
) -> ExperimentData:
    x = torch.linspace(-3, 3, number_of_data_points).reshape(-1, 1)
    y_curve = 2 * curve_function.calculate_curve(
        x=x,
    ).reshape(-1)
    bernoulli_noise = torch.bernoulli(
        input=torch.tensor([bernoulli_probability_true]),
        generator=torch.Generator().manual_seed(curve_function.seed),
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
        normalise=False,
    )
    experiment_data.full.y_untransformed = (
        y_curve + bernoulli_shift_true * bernoulli_noise
    )
    # experiment_data.train.y_untransformed = experiment_data.train.y.clone()
    # experiment_data.validation.y_untransformed = experiment_data.validation.y.clone()
    # experiment_data.test.y_untransformed = experiment_data.test.y.clone()

    # experiment_data.full.y += bernoulli_shift_true * bernoulli_noise
    # experiment_data.train.y += bernoulli_shift_true * bernoulli_noise
    # experiment_data.validation.y += bernoulli_shift_true * bernoulli_noise
    # experiment_data.test.y += bernoulli_shift_true * bernoulli_noise
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


def generate_init_particles(
    initial_particle_noise: float,
    approximation_dimension: int,
    number_of_particles: int,
    initial_particles_lower: float,
    initial_particles_shift_scale: float,
    bernoulli_shift_true: float,
    basis_dimension: int,
    basis_eigenvectors: torch.Tensor,
    basis_eigenvalues: torch.Tensor,
) -> torch.Tensor:
    init_particles = torch.normal(
        0,
        initial_particle_noise,
        size=(approximation_dimension, number_of_particles),
    )
    init_particles += torch.linspace(
        initial_particles_lower, initial_particles_shift_scale * bernoulli_shift_true, number_of_particles
    )[None, :]
    return (
        math.sqrt(basis_dimension)
        * basis_eigenvectors.T
        @ torch.diag(torch.divide(1, torch.sqrt(basis_eigenvalues)))
        @ init_particles
    )


def main(
    curve_function: Curve,
    data_config: Dict[str, Any],
    kernel_config: Dict[str, Any],
    inducing_points_config: Dict[str, Any],
    pls_config: Dict[str, Any],
    include_gif: bool,
    data_path: str,
    plot_curve_path: str,
    models_path: str,
) -> None:
    experiment_data = get_experiment_data(
        curve_function=curve_function,
        number_of_data_points=data_config["number_of_data_points"],
        seed=data_config["seed"],
        bernoulli_probability_true=data_config["bernoulli_probability_true"],
        bernoulli_shift_true=data_config["bernoulli_shift_true"],
        sigma_true=data_config["sigma_true"],
        train_data_percentage=data_config["train_data_percentage"],
        validation_data_percentage=data_config["validation_data_percentage"],
    )
    plot_experiment_data(
        experiment_data=experiment_data,
        title=f"{curve_function.__name__} data",
        plot_curve_path=plot_curve_path,
    )
    plot_experiment_data(
        experiment_data=experiment_data,
        title=f"{curve_function.__name__} data GP",
        plot_curve_path=plot_curve_path,
    )
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
        model_path=os.path.join(models_path, "subsample_gp"),
        data_path=os.path.join(data_path, "subsample_gp"),
        plot_1d_subsample_path=None,
        plot_loss_path=plot_curve_path,
    )
    average_ard_kernel = construct_average_ard_kernel(
        kernels=[model.kernel for model in subsample_gp_models]
    )
    plot_gp_prediction(
        experiment_data=experiment_data,
        model=subsample_gp_models[0],
        title=f"{curve_function.__name__} Exact GP",
        save_path=os.path.join(plot_curve_path, "exact_gp.png"),
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
    cost = MultiModalCost(
        observation_noise=data_config["sigma_true"],
        y_train=experiment_data.train.y,
        link_function=IdentityLinkFunction(),
        shift=data_config["bernoulli_shift_true"],
        bernoulli_noise=data_config["bernoulli_probability_true"],
    )
    plot_title = "PLS for Multi-modal Regression"
    pls = ProjectedLangevinSampling(basis=onb_basis, cost=cost, name="pls-onb")
    set_seed(pls_config["seed"])
    init_particles = generate_init_particles(
        initial_particle_noise=pls_config["initial_particle_noise"],
        approximation_dimension=onb_basis.approximation_dimension,
        number_of_particles=pls_config["number_of_particles"],
        initial_particles_lower=pls_config["initial_particles_lower"],
        initial_particles_shift_scale=pls_config["initial_particles_shift_scale"],
        bernoulli_shift_true=data_config["bernoulli_shift_true"],
        basis_dimension=onb_basis.x_induce.shape[0],
        basis_eigenvectors=onb_basis.eigenvectors,
        basis_eigenvalues=onb_basis.eigenvalues,
    )
    plot_pls_1d_particles_runner(
        pls=pls,
        particles=init_particles.clone(),
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
    particles, _, _ = train_pls_runner(
        pls=pls,
        particles=init_particles.clone(),
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
    plot_pls_1d_particles_runner(
        pls=pls,
        particles=particles,
        particle_name=f"{pls.name}-learned",
        experiment_data=experiment_data,
        inducing_points=inducing_points,
        plot_particles_path=plot_curve_path,
        plot_title=plot_title,
    )
    if include_gif:
        set_seed(pls_config["seed"])
        init_particles = generate_init_particles(
            initial_particle_noise=pls_config["initial_particle_noise"],
            approximation_dimension=onb_basis.approximation_dimension,
            number_of_particles=pls_config["number_of_particles"],
            initial_particles_lower=pls_config["initial_particles_lower"],
            initial_particles_shift_scale=pls_config[
                "gif_initial_particles_shift_scale"
            ],
            bernoulli_shift_true=data_config["bernoulli_shift_true"],
            basis_dimension=onb_basis.x_induce.shape[0],
            basis_eigenvectors=onb_basis.eigenvectors,
            basis_eigenvalues=onb_basis.eigenvalues,
        )
        animate_pls_1d_particles_runner(
            pls=pls,
            number_of_particles=pls_config["number_of_particles"],
            particle_name=pls.name,
            experiment_data=experiment_data,
            seed=pls_config["seed"],
            best_lr=pls_config["gif_lr"],
            number_of_epochs=pls_config["gif_number_of_epochs"],
            plot_title=plot_title,
            animate_1d_path=plot_curve_path,
            animate_1d_untransformed_path=plot_curve_path,
            christmas_colours=pls_config["christmas_colours"]
            if "christmas_colours" in pls_config
            else False,
            initial_particles_noise_only=pls_config["initial_particles_noise_only"],
            init_particles=init_particles.clone(),
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
            include_gif=args.include_gif,
            data_path=os.path.join(
                OUTPUTS_PATH, "data", type(curve_function_).__name__.lower()
            ),
            plot_curve_path=os.path.join(
                OUTPUTS_PATH, "plots", type(curve_function_).__name__.lower()
            ),
            models_path=os.path.join(
                OUTPUTS_PATH, "models", type(curve_function_).__name__.lower()
            ),
        )
