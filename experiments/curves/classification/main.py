import argparse
import math
import os
from copy import deepcopy
from typing import Any, Dict

import gpytorch
import matplotlib.pyplot as plt
import torch
import yaml

from experiments.constructors import construct_average_ard_kernel
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
from experiments.utils import create_directory
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.projected_langevin_sampling import PLS, PLSKernel
from src.projected_langevin_sampling.basis import InducingPointBasis, OrthonormalBasis
from src.projected_langevin_sampling.costs import BernoulliCost
from src.projected_langevin_sampling.link_functions import (
    ProbitLinkFunction,
    SigmoidLinkFunction,
)
from src.utils import set_seed

parser = argparse.ArgumentParser(
    description="Main script for toy binary classification experiments."
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
    number_of_test_intervals: int,
    total_number_of_intervals: int,
) -> ExperimentData:
    x = torch.linspace(-3, 3, number_of_data_points).reshape(-1, 1)
    y_curve = 2 * curve_function.calculate_curve(
        x=x,
    ).reshape(-1)
    y = curve_function.classification(
        y_curve=y_curve,
        seed=seed,
    )
    link_function = SigmoidLinkFunction()
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
        problem_type=ProblemType.CLASSIFICATION,
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
    ax.legend()
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
    likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(
        experiment_data.train.y, learn_additional_noise=True
    )
    y_train_labels = experiment_data.train.y
    experiment_data.train.y = likelihood.transformed_targets
    subsample_gp_models = exact_gp_runner(
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
        early_stopper_patience=kernel_config["early_stopper_patience"],
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
    sigmoid_cost = BernoulliCost(
        y_train=experiment_data.train.y,
        link_function=SigmoidLinkFunction(),
    )
    probit_cost = BernoulliCost(
        y_train=experiment_data.train.y,
        link_function=ProbitLinkFunction(),
    )
    pls_dict = {
        "pls-onb-probit": PLS(
            basis=onb_basis,
            cost=probit_cost,
        ),
        "pls-ipb-probit": PLS(
            basis=ipb_basis,
            cost=probit_cost,
        ),
        "pls-onb-sigmoid": PLS(
            basis=onb_basis,
            cost=sigmoid_cost,
        ),
        "pls-ipb-sigmoid": PLS(
            basis=ipb_basis,
            cost=sigmoid_cost,
        ),
    }
    plot_title = "PLS for Binary Classification"
    for pls_name, pls in pls_dict.items():
        pls_path = os.path.join(models_path, f"{pls_name}.pth")
        set_seed(pls_config["seed"])
        particles = pls.initialise_particles(
            number_of_particles=pls_config["number_of_particles"],
            seed=pls_config["seed"],
            noise_only=pls_config["initial_particles_noise_only"],
        )
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
            animate_1d_path=plot_curve_path,
            plot_title=plot_title,
            animate_1d_untransformed_path=None,
            christmas_colours=pls_config["christmas_colours"]
            if "christmas_colours" in pls_config
            else False,
            initial_particles_noise_only=pls_config["initial_particles_noise_only"],
        )

    plot_title = "SVGP for Binary Classification"
    svgp_likelihood = gpytorch.likelihoods.BernoulliLikelihood()
    for kernel_name, kernel in zip(["k", "r"], [average_ard_kernel, pls_kernel]):
        model_name = f"svgp-{kernel_name}"
        svgp_model_path = os.path.join(models_path, f"{model_name}.pth")
        if os.path.exists(svgp_model_path):
            svgp, losses, best_learning_rate = load_svgp(
                model_path=svgp_model_path,
                x_induce=inducing_points.x,
                mean=gpytorch.means.ConstantMean(),
                kernel=deepcopy(kernel),
                likelihood=gpytorch.likelihoods.BernoulliLikelihood(),
                learn_inducing_locations=False,
            )
        else:
            svgp, losses, best_learning_rate = train_svgp_runner(
                model_name=model_name,
                experiment_data=experiment_data,
                inducing_points=inducing_points,
                mean=gpytorch.means.ConstantMean(),
                kernel=deepcopy(kernel),
                likelihood=svgp_likelihood,
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
                early_stopper_patience=svgp_config["early_stopper_patience"],
                models_path=os.path.join(
                    models_path, f"{model_name}-kernel-iterations"
                ),
                plot_title=plot_title,
                plot_loss_path=plot_curve_path,
            )
            torch.save(
                {
                    "model": svgp.state_dict(),
                    "losses": losses,
                    "best_learning_rate": best_learning_rate,
                },
                svgp_model_path,
            )
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
        animate_1d_gp_predictions(
            experiment_data=experiment_data,
            inducing_points=inducing_points,
            mean=deepcopy(svgp.mean),
            kernel=deepcopy(svgp.kernel),
            likelihood=deepcopy(svgp_likelihood),
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
