import argparse
import math
import os
from copy import deepcopy
from typing import Any, Dict

import gpytorch
import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.model_selection import train_test_split

from experiments.constructors import (
    construct_average_ard_kernel,
    construct_average_gaussian_likelihood,
)
from experiments.curves.curves import CURVE_FUNCTIONS, Curve
from experiments.data import Data, ExperimentData, ProblemType
from experiments.loaders import load_pls, load_svgp
from experiments.plotters import (
    animate_1d_gp_predictions,
    plot_1d_experiment_data,
    plot_1d_gp_prediction_and_inducing_points,
)
from experiments.preprocess import split_regression_data_intervals
from experiments.runners import (
    animate_pls_1d_particles_runner,
    exact_gp_runner,
    inducing_points_runner,
    plot_pls_1d_particles_runner,
    train_pls_runner,
    train_svgp_runner,
)
from experiments.utils import create_directory, str2bool
from src.conformalise import ConformaliseGP, ConformalisePLS
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.kernels.projected_langevin_sampling import PLSKernel
from src.projected_langevin_sampling import ProjectedLangevinSampling
from src.projected_langevin_sampling.basis import OrthonormalBasis
from src.projected_langevin_sampling.costs import GaussianCost
from src.projected_langevin_sampling.costs.student_t import StudentTCost
from src.projected_langevin_sampling.link_functions import IdentityLinkFunction

# from src.temper import TemperGP, TemperPLS
from src.utils import set_seed

parser = argparse.ArgumentParser(
    description="Main script for toy Poisson regression experiments with out of distribution test samples."
)
parser.add_argument("--config_path", type=str)
parser.add_argument(
    "--include_gif",
    type=str2bool,
    default=False,
    help="Indicate whether to include GIFs in the output.",
)


def get_experiment_data(
    curve_function: Curve,
    number_of_data_points: int,
    seed: int,
    degrees_of_freedom: float,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
    validation_data_percentage: float,
    min_validation_data_points: int = 50,
) -> ExperimentData:
    x = torch.linspace(-3, 3, number_of_data_points).reshape(-1, 1)
    y_curve = 2 * curve_function.calculate_curve(
        x=x,
    ).reshape(-1)
    link_function = IdentityLinkFunction()
    set_seed(seed)
    y = link_function.transform(y_curve) + torch.distributions.studentT.StudentT(
        loc=0.0,
        df=degrees_of_freedom,
    ).sample(sample_shape=y_curve.shape)
    (
        x_train_validation,
        y_train_validation,
        _,
        x_test,
        y_test,
        _,
    ) = split_regression_data_intervals(
        split_seed=curve_function.seed,
        x=x,
        y=y,
        number_of_test_intervals=number_of_test_intervals,
        total_number_of_intervals=total_number_of_intervals,
    )
    if len(x_train_validation) < min_validation_data_points:
        raise ValueError(
            f"Number of training points is less than the minimum number of validation points: {len(x_train_validation)} < {min_validation_data_points}"
        )
    (
        x_train,
        x_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        x_train_validation,
        y_train_validation,
        test_size=max(
            validation_data_percentage,
            min_validation_data_points / len(x_train_validation),
        ),
        random_state=seed,
    )
    experiment_data = ExperimentData(
        name=type(curve_function).__name__.lower(),
        problem_type=ProblemType.REGRESSION,
        full=Data(x=x, y=y, name="full"),
        train=Data(x=x_train, y=y_train, name="train"),
        validation=Data(x=x_validation, y=y_validation, name="validation"),
        test=Data(x=x_test, y=y_test, name="test"),
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
    svgp_config: Dict[str, Any],
    outputs_path: str,
    include_gif: bool,
) -> None:
    experiment_data = get_experiment_data(
        curve_function=curve_function,
        number_of_data_points=data_config["number_of_data_points"],
        seed=data_config["seed"],
        degrees_of_freedom=data_config["degrees_of_freedom"],
        number_of_test_intervals=data_config["number_of_test_intervals"],
        total_number_of_intervals=data_config["total_number_of_intervals"],
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
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    subsample_gp_models = exact_gp_runner(
        experiment_data=experiment_data,
        kernel=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=experiment_data.train.x.shape[1])
        ),
        likelihood=likelihood,
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
    degrees_of_freedom = float(2 * likelihood.noise) / float(likelihood.noise - 1)
    if degrees_of_freedom <= 2:
        print(
            f"Degrees of freedom is {degrees_of_freedom} which is less than or equal to 2. Skipping Student T cost."
        )
        return
    t_noise_distribution = torch.distributions.studentT.StudentT(
        loc=0.0,
        scale=likelihood.noise,
        df=degrees_of_freedom,
    )
    pls_kernel = PLSKernel(
        base_kernel=average_ard_kernel,
        approximation_samples=inducing_points.x,
    )
    onb_basis = OrthonormalBasis(
        kernel=pls_kernel,
        x_induce=inducing_points.x,
        x_train=experiment_data.train.x,
        additional_predictive_noise_distribution=t_noise_distribution,
    )
    cost = StudentTCost(
        degrees_of_freedom=degrees_of_freedom,
        y_train=experiment_data.train.y,
        link_function=IdentityLinkFunction(),
    )
    plot_title = "PLS for Student T Regression"
    pls = ProjectedLangevinSampling(basis=onb_basis, cost=cost, name="pls-onb")
    pls_path = os.path.join(models_path, f"{pls.name}.pth")
    set_seed(pls_config["seed"])
    particles = pls.initialise_particles(
        number_of_particles=pls_config["number_of_particles"],
        seed=pls_config["seed"],
        noise_only=pls_config["initial_particles_noise_only"],
    )
    plot_pls_1d_particles_runner(
        pls=pls,
        particles=particles[:, :10],
        particle_name=f"{pls.name}-initial",
        experiment_data=experiment_data,
        plot_particles_path=plot_curve_path,
        plot_title=plot_title,
        number_of_particles_to_plot=10,
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
    pls_conformalised = ConformalisePLS(
        x_calibration=experiment_data.validation.x,
        y_calibration=experiment_data.validation.y,
        pls=pls,
        particles=particles,
    )
    # pls_tempered = TemperPLS(
    #     x_calibration=experiment_data.validation.x,
    #     y_calibration=experiment_data.validation.y,
    #     pls=pls,
    #     particles=particles,
    # )
    plot_pls_1d_particles_runner(
        pls=pls,
        particles=particles,
        particle_name=f"{pls.name}-learned",
        experiment_data=experiment_data,
        plot_particles_path=plot_curve_path,
        plot_title=plot_title,
        number_of_particles_to_plot=10,
    )
    plot_pls_1d_particles_runner(
        pls=pls_conformalised,
        particles=particles,
        particle_name=f"{pls.name}-learned-conformalised",
        experiment_data=experiment_data,
        plot_particles_path=plot_curve_path,
        plot_title=f"{plot_title} Conformalised",
    )
    # plot_pls_1d_particles_runner(
    #     pls=pls_tempered,
    #     particles=particles,
    #     particle_name=f"{pls.name}-learned-tempered",
    #     experiment_data=experiment_data,
    #     plot_particles_path=plot_curve_path,
    #     plot_title=f"{plot_title} Tempered",
    # )
    if include_gif:
        animate_pls_1d_particles_runner(
            pls=pls,
            number_of_particles=10,
            particle_name=pls.name,
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

    plot_title = "SVGP for Regression"
    model_name = f"svgp-r"
    svgp_model_path = os.path.join(models_path, f"{model_name}.pth")
    if os.path.exists(svgp_model_path):
        svgp, losses, best_learning_rate = load_svgp(
            model_path=svgp_model_path,
            x_induce=inducing_points.x,
            mean=gpytorch.means.ConstantMean(),
            kernel=deepcopy(pls_kernel),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            learn_inducing_locations=False,
        )
    else:
        svgp, losses, best_learning_rate = train_svgp_runner(
            model_name=model_name,
            experiment_data=experiment_data,
            inducing_points=inducing_points,
            mean=gpytorch.means.ConstantMean(),
            kernel=deepcopy(pls_kernel),
            likelihood=gpytorch.likelihoods.GaussianLikelihood(),
            seed=svgp_config["seed"],
            number_of_epochs=svgp_config["number_of_epochs"],
            batch_size=svgp_config["batch_size"],
            learning_rate_upper=svgp_config["learning_rate_upper"],
            learning_rate_lower=svgp_config["learning_rate_lower"],
            number_of_learning_rate_searches=svgp_config[
                "number_of_learning_rate_searches"
            ],
            is_fixed=True,
            observation_noise=float(likelihood.noise),
            early_stopper_patience=svgp_config["early_stopper_patience"],
            models_path=os.path.join(models_path, f"{model_name}-kernel-iterations"),
            plot_title=plot_title,
            plot_loss_path=plot_curve_path,
        )
        torch.save(
            {
                "model": svgp.state_dict(),
                "losses": losses,
                "best_learning_rate": best_learning_rate,
            },
            os.path.join(models_path, f"{model_name}.pth"),
        )
    svgp_conformalised = ConformaliseGP(
        x_calibration=experiment_data.validation.x,
        y_calibration=experiment_data.validation.y,
        gp=svgp,
    )
    # svgp_tempered = TemperGP(
    #     x_calibration=experiment_data.validation.x,
    #     y_calibration=experiment_data.validation.y,
    #     gp=svgp,
    # )
    plot_1d_gp_prediction_and_inducing_points(
        model=svgp,
        experiment_data=experiment_data,
        inducing_points=inducing_points,
        title=plot_title,
        save_path=os.path.join(
            plot_curve_path,
            f"{model_name}.png",
        ),
    )
    plot_1d_gp_prediction_and_inducing_points(
        model=svgp_conformalised,
        experiment_data=experiment_data,
        inducing_points=inducing_points,
        title=f"{plot_title} Conformalised",
        save_path=os.path.join(
            plot_curve_path,
            f"{model_name}-conformalised.png",
        ),
    )
    # plot_1d_gp_prediction_and_inducing_points(
    #     model=svgp_tempered,
    #     experiment_data=experiment_data,
    #     inducing_points=inducing_points,
    #     title=f"{plot_title} Tempered",
    #     save_path=os.path.join(
    #         plot_curve_path,
    #         f"{model_name}-tempered.png",
    #     ),
    # )
    if include_gif:
        animate_1d_gp_predictions(
            experiment_data=experiment_data,
            inducing_points=inducing_points,
            mean=deepcopy(svgp.mean),
            kernel=deepcopy(svgp.kernel),
            likelihood=deepcopy(likelihood),
            seed=svgp_config["seed"],
            number_of_epochs=len(losses),
            batch_size=svgp_config["batch_size"],
            learning_rate=best_learning_rate,
            title=plot_title,
            save_path=os.path.join(
                plot_curve_path,
                f"{model_name}.gif",
            ),
            learn_inducing_locations=False,
            learn_kernel_parameters=False,
            christmas_colours=svgp_config["christmas_colours"]
            if "christmas_colours" in pls_config
            else False,
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
            svgp_config=loaded_config["svgp"],
            outputs_path=outputs_path,
            include_gif=args.include_gif,
        )