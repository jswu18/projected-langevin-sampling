import argparse
import math
import os
from copy import deepcopy
from typing import Any, Dict

import gpytorch
import matplotlib.pyplot as plt
import scipy
import torch
import yaml

from experiments.constructors import (
    construct_average_ard_kernel,
    construct_average_gaussian_likelihood,
)
from experiments.curves.curves import CURVE_FUNCTIONS, Curve
from experiments.data import ExperimentData, ProblemType
from experiments.loaders import load_pls, load_svgp
from experiments.metrics import calculate_metrics, concatenate_metrics
from experiments.plotters import (
    animate_1d_gp_predictions,
    plot_1d_conformal_prediction,
    plot_1d_experiment_data,
    plot_1d_gp_prediction_and_inducing_points,
)
from experiments.preprocess import set_up_experiment
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
    description="Main script for toy Poisson regression experiments."
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
    train_data_percentage: float,
    validation_data_percentage: float,
) -> ExperimentData:
    x = torch.linspace(-3, 3, number_of_data_points).reshape(-1, 1)
    y_curve = 2 * curve_function.calculate_curve(
        x=x,
    ).reshape(-1)
    link_function = IdentityLinkFunction()
    set_seed(seed)
    y = link_function.transform(y_curve) + torch.distributions.studentT.StudentT(
        loc=0.0,
        scale=1.0,
        df=degrees_of_freedom,
    ).sample(sample_shape=y_curve.shape)
    experiment_data = set_up_experiment(
        name=curve_function.__name__,
        problem_type=ProblemType.REGRESSION,
        seed=seed,
        x=x,
        y=y,
        train_data_percentage=train_data_percentage,
        validation_data_percentage=validation_data_percentage,
        normalise=True,
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
    metrics_config: Dict[str, Any],
    outputs_path: str,
    include_gif: bool,
) -> None:
    experiment_data = get_experiment_data(
        curve_function=curve_function,
        number_of_data_points=data_config["number_of_data_points"],
        seed=data_config["seed"],
        degrees_of_freedom=data_config["degrees_of_freedom"],
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
    # model_name = "exact-gp"
    # plot_title = "Student T SVGP with All Data"
    # exact_svgp, _, _ = train_svgp_runner(
    #     model_name=model_name,
    #     experiment_data=experiment_data,
    #     inducing_points=experiment_data.train,
    #     mean=gpytorch.means.ConstantMean(),
    #     kernel=gpytorch.kernels.ScaleKernel(
    #         gpytorch.kernels.RBFKernel(ard_num_dims=experiment_data.train.x.shape[1])
    #     ),
    #     likelihood=gpytorch.likelihoods.StudentTLikelihood(),
    #     seed=kernel_config["seed"],
    #     number_of_epochs=svgp_config["number_of_epochs"],
    #     batch_size=svgp_config["batch_size"],
    #     learning_rate_upper=kernel_config["learning_rate"],
    #     learning_rate_lower=kernel_config["learning_rate"],
    #     number_of_learning_rate_searches=1,
    #     is_fixed=False,
    #     observation_noise=1.0,
    #     early_stopper_patience=kernel_config["early_stopper_patience"],
    #     models_path=os.path.join(models_path, f"{model_name}-kernel-exact"),
    #     plot_title=plot_title,
    #     plot_loss_path=plot_curve_path,
    # )
    # average_ard_kernel = exact_svgp.kernel
    # plot_1d_gp_prediction_and_inducing_points(
    #     model=exact_svgp,
    #     experiment_data=experiment_data,
    #     inducing_points=None,
    #     title=f"{plot_title} SVGP Exact",
    #     save_path=os.path.join(
    #         plot_curve_path,
    #         f"{model_name}-svgp-exact.png",
    #     ),
    # )
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
    for i, exact_gp_iter in enumerate(subsample_gp_models):
        plot_1d_gp_prediction_and_inducing_points(
            model=exact_gp_iter,
            experiment_data=experiment_data,
            inducing_points=None,
            title=f"Subsample GP {i}" if len(subsample_gp_models) > 1 else "Exact GP",
            save_path=os.path.join(
                plot_curve_path,
                f"subsample-gp-{i}.png"
                if len(subsample_gp_models) > 1
                else "exact-gp.png",
            ),
            coverage=metrics_config["coverage"],
        )
    exact_gp_predictions = [
        model.likelihood(model(experiment_data.train.x))
        for model in subsample_gp_models
    ]
    residuals = (
        torch.stack(
            (
                [
                    experiment_data.train.y - prediction.mean
                    for prediction in exact_gp_predictions
                ]
            ),
            axis=1,
        )
        .mean(
            axis=1,
        )
        .cpu()
        .detach()
        .numpy()
    )
    degrees_of_freedom, _, _ = scipy.stats.t.fit(residuals, floc=0)
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
    # degrees_of_freedom = float(2 * likelihood.noise) / float(likelihood.noise - 1)
    # if degrees_of_freedom <= 2:
    #     print(
    #         f"The calculated degrees of freedom is {degrees_of_freedom} which is less than or equal to 2. The noise is {float(likelihood.noise)}. Setting degrees of freedom to 100.0."
    #     )
    #     degrees_of_freedom = 3.0
    # t_noise_distribution = torch.distributions.studentT.StudentT(
    #     loc=0.0,
    #     scale=likelihood.noise,
    #     df=degrees_of_freedom,
    # )
    # degrees_of_freedom = int(exact_svgp.likelihood.deg_free)
    t_noise_distribution = torch.distributions.studentT.StudentT(
        loc=0.0,
        scale=likelihood.noise,
        df=degrees_of_freedom,
    )
    print(f"Degrees of freedom: {degrees_of_freedom}")
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
        coverage=metrics_config["coverage"],
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
        coverage=metrics_config["coverage"],
    )
    plot_1d_conformal_prediction(
        model=pls_conformalised,
        experiment_data=experiment_data,
        plot_title=f"{plot_title} Conformalised",
        coverage=metrics_config["coverage"],
        save_path=os.path.join(
            plot_curve_path,
            f"{pls.name}-conformalised.png",
        ),
    )
    set_seed(pls_config["seed"])
    calculate_metrics(
        model=pls_conformalised,
        particles=particles,
        model_name="pls-onb-conformalised",
        dataset_name=type(curve_function).__name__.lower(),
        experiment_data=experiment_data,
        results_path=results_path,
        plots_path=plot_curve_path,
        coverage=metrics_config["coverage"],
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
            likelihood=gpytorch.likelihoods.StudentTLikelihood(),
            learn_inducing_locations=False,
        )
    else:
        svgp, losses, best_learning_rate = train_svgp_runner(
            model_name=model_name,
            experiment_data=experiment_data,
            inducing_points=inducing_points,
            mean=gpytorch.means.ConstantMean(),
            kernel=deepcopy(pls_kernel),
            likelihood=gpytorch.likelihoods.StudentTLikelihood(),
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
        coverage=metrics_config["coverage"],
    )
    plot_1d_conformal_prediction(
        model=svgp_conformalised,
        experiment_data=experiment_data,
        plot_title=f"{plot_title} Conformalised",
        coverage=metrics_config["coverage"],
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
    set_seed(svgp_config["seed"])
    calculate_metrics(
        model=svgp_conformalised,
        experiment_data=experiment_data,
        model_name="svgp-conformalised",
        dataset_name=type(curve_function).__name__.lower(),
        results_path=results_path,
        plots_path=plot_curve_path,
        coverage=metrics_config["coverage"],
    )
    if include_gif:
        animate_1d_gp_predictions(
            experiment_data=experiment_data,
            inducing_points=inducing_points,
            mean=deepcopy(svgp.mean),
            kernel=deepcopy(svgp.kernel),
            likelihood=gpytorch.likelihoods.StudentTLikelihood(),
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
            metrics_config=loaded_config["metrics"],
            outputs_path=outputs_path,
            include_gif=args.include_gif,
        )
    concatenate_metrics(
        results_path=os.path.join(outputs_path, "results"),
        data_types=["train", "test"],
        model_names=[
            "pls-onb-conformalised",
            "svgp-conformalised",
        ],
        datasets=[
            type(curve_function_).__name__.lower()
            for curve_function_ in CURVE_FUNCTIONS
        ],
        metrics=[
            "mae",
            "mse",
            "nll",
            "average_interval_width",
            "median_interval_width",
            "coverage",
        ],
    )
