import argparse
import math
import os
from copy import deepcopy
from typing import Any, Dict, Tuple

import gpytorch
import pandas as pd
import scipy
import torch
import yaml

from experiments.constructors import (
    construct_average_ard_kernel,
    construct_average_gaussian_likelihood,
)
from experiments.data import ExperimentData, ProblemType
from experiments.loaders import load_pls, load_svgp
from experiments.metrics import calculate_metrics, concatenate_metrics
from experiments.plotters import plot_eigenvalues
from experiments.preprocess import set_up_experiment
from experiments.runners import (
    exact_gp_runner,
    inducing_points_runner,
    train_pls_runner,
    train_svgp_runner,
)
from experiments.uci.constants import DATASET_SCHEMA_MAPPING, RegressionDatasetSchema
from src.conformalise import ConformalisePLS
from src.conformalise.gp import ConformaliseGP
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.projected_langevin_sampling import PLS, PLSKernel
from src.projected_langevin_sampling.basis import OrthonormalBasis
from src.projected_langevin_sampling.costs import GaussianCost
from src.projected_langevin_sampling.costs.student_t import StudentTCost
from src.projected_langevin_sampling.link_functions import IdentityLinkFunction
from src.temper import TemperGP, TemperPLS
from src.utils import set_seed

parser = argparse.ArgumentParser(
    description="Main script for UCI regression data experiments."
)
parser.add_argument(
    "--config_path", type=str, required=True, help="Path to config for experiment."
)
parser.add_argument(
    "--data_seed",
    type=int,
    required=False,
    default=-1,
    help="Seed to use for the data split of the experiment.",
)

MODEL_NAMES = [
    "pls-onb",
    "pls-onb-temper",
    "pls-onb-conformalise",
    "pls-student-onb",
    "pls-student-onb-temper",
    "pls-student-onb-conformalise",
    "svgp",
    "svgp-temper",
    "svgp-conformalise",
    "svgp-student",
    "svgp-student-temper",
    "svgp-student-conformalise",
]


METRICS = ["mae", "mse", "nll", "average_interval_width", "coverage"]


def get_experiment_data(
    seed: int,
    train_data_percentage: float,
    validation_data_percentage: float,
    dataset_name: str,
) -> ExperimentData:
    assert train_data_percentage + validation_data_percentage <= 1.0, (
        f"{train_data_percentage=} and {validation_data_percentage=} "
        "should sum to less than or equal to 1.0."
    )
    df = pd.read_csv(
        os.path.join("experiments", "uci", "datasets", f"{dataset_name}.csv")
    )
    df.columns = [c.lower() for c in df.columns]
    df.columns = [c.replace(" ", "") for c in df.columns]
    dataset_metadata = DATASET_SCHEMA_MAPPING[dataset_name]
    input_column_names = [c.lower() for c in dataset_metadata.input_column_names]
    input_column_names = [c.replace(" ", "") for c in input_column_names]
    output_column_name = dataset_metadata.output_column_name.lower().replace(" ", "")

    x = torch.tensor(df[input_column_names].to_numpy()).detach().double()
    y = torch.tensor(df[output_column_name].to_numpy()).detach().double()

    experiment_data = set_up_experiment(
        name=dataset_name,
        problem_type=ProblemType.REGRESSION,
        seed=seed,
        x=x,
        y=y,
        train_data_percentage=train_data_percentage,
        validation_data_percentage=validation_data_percentage,
        normalise=True,
    )
    return experiment_data


def estimate_student_parameters(
    y_actual: torch.Tensor, predictions: list[torch.distributions.Distribution]
) -> Tuple[float, float]:
    residuals = (
        torch.stack(
            ([y_actual - prediction.mean for prediction in predictions]),
            axis=1,
        )
        .mean(
            axis=1,
        )
        .cpu()
        .detach()
        .numpy()
    )
    degrees_of_freedom, _, scale = scipy.stats.t.fit(residuals, floc=0)
    return degrees_of_freedom, scale


def main(
    data_seed: int,
    dataset_name: str,
    data_config: Dict[str, Any],
    kernel_config: Dict[str, Any],
    inducing_points_config: Dict[str, Any],
    pls_config: Dict[str, Any],
    svgp_config: Dict[str, Any],
    metrics_config: Dict[str, Any],
    outputs_path: str,
) -> None:
    print(f"Running experiment for {dataset_name=} and {data_seed=}.")

    data_path = os.path.join(outputs_path, str(data_seed), "data", dataset_name)
    plots_path = os.path.join(outputs_path, str(data_seed), "plots", dataset_name)
    results_path = os.path.join(outputs_path, str(data_seed), "results", dataset_name)
    models_path = os.path.join(outputs_path, str(data_seed), "models", dataset_name)
    subsample_gp_model_path = os.path.join(models_path, "subsample_gp")
    subsample_gp_data_path = os.path.join(data_path, "subsample_gp")
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    experiment_data_path = os.path.join(data_path, "experiment_data.pth")
    inducing_points_path = os.path.join(data_path, "inducing_points.pth")

    if os.path.exists(experiment_data_path):
        experiment_data = ExperimentData.load(
            path=experiment_data_path, problem_type=ProblemType.REGRESSION
        )
        print(f"Loaded experiment data from {experiment_data_path=}")
    else:
        experiment_data = get_experiment_data(
            seed=data_seed,
            train_data_percentage=data_config["train_data_percentage"],
            validation_data_percentage=data_config["validation_data_percentage"],
            dataset_name=dataset_name,
        )
        experiment_data.save(experiment_data_path)

    subsample_gp_models = exact_gp_runner(
        experiment_data=experiment_data,
        kernel=gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=experiment_data.train.x.shape[1])
        ),
        subsample_size=kernel_config["subsample_size"],
        seed=kernel_config["seed"],
        number_of_epochs=kernel_config["number_of_epochs"],
        learning_rate=kernel_config["learning_rate"],
        number_of_iterations=kernel_config["number_of_iterations"],
        early_stopper_patience=kernel_config["early_stopper_patience"],
        plot_1d_subsample_path=None,
        plot_loss_path=plots_path,
        model_path=subsample_gp_model_path,
        data_path=subsample_gp_data_path,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    )
    average_ard_kernel = construct_average_ard_kernel(
        kernels=[model.kernel for model in subsample_gp_models]
    )
    if os.path.exists(inducing_points_path):
        inducing_points = torch.load(inducing_points_path)
    else:
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
        torch.save(inducing_points, inducing_points_path)
    likelihood = construct_average_gaussian_likelihood(
        likelihoods=[model.likelihood for model in subsample_gp_models]
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
    gaussian_cost = GaussianCost(
        observation_noise=float(likelihood.noise),
        y_train=experiment_data.train.y,
        link_function=IdentityLinkFunction(),
    )

    for model in subsample_gp_models:
        model.eval()
        model.likelihood.eval()
    degrees_of_freedom, scale = estimate_student_parameters(
        y_actual=torch.Tensor(experiment_data.train.y),
        predictions=[
            model.likelihood(model(experiment_data.train.x))
            for model in subsample_gp_models
        ],
    )
    t_noise_distribution = torch.distributions.studentT.StudentT(
        loc=0.0,
        scale=likelihood.noise,
        df=degrees_of_freedom,
    )
    student_onb_basis = OrthonormalBasis(
        kernel=pls_kernel,
        x_induce=inducing_points.x,
        x_train=experiment_data.train.x,
        additional_predictive_noise_distribution=t_noise_distribution,
    )
    student_cost = StudentTCost(
        degrees_of_freedom=degrees_of_freedom,
        y_train=experiment_data.train.y,
        link_function=IdentityLinkFunction(),
        scale=scale,
    )
    pls_dict = {
        "pls-onb": PLS(
            basis=onb_basis,
            cost=gaussian_cost,
        ),
        "pls-student-onb": PLS(
            basis=student_onb_basis,
            cost=student_cost,
        ),
    }
    for pls_name, pls in pls_dict.items():
        if isinstance(pls.basis, OrthonormalBasis):
            plot_eigenvalues(
                basis=pls.basis,
                save_path=os.path.join(plots_path, f"eigenvalues-{pls_name}.png"),
                title=f"Eigenvalues ({dataset_name})",
            )
        pls_path = os.path.join(models_path, f"{pls_name}.pth")
        set_seed(pls_config["seed"])
        particles = pls.initialise_particles(
            number_of_particles=pls_config["number_of_particles"],
            seed=pls_config["seed"],
            noise_only=pls_config["initial_particles_noise_only"],
        )
        if os.path.exists(pls_path):
            pls, particles, _, _ = load_pls(
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
                maximum_number_of_steps=pls_config["maximum_number_of_steps"],
                step_size_upper=pls_config["step_size_upper"],
                number_of_step_searches=pls_config["number_of_step_searches"],
                minimum_change_in_energy_potential=pls_config[
                    "minimum_change_in_energy_potential"
                ],
                seed=pls_config["seed"],
                plot_title=f"{dataset_name}",
                plot_energy_potential_path=plots_path,
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
        set_seed(pls_config["seed"])
        calculate_metrics(
            model=pls,
            particles=particles,
            model_name=pls_name,
            dataset_name=dataset_name,
            experiment_data=experiment_data,
            results_path=results_path,
            plots_path=plots_path,
            coverage=metrics_config["coverage"],
        )
        set_seed(pls_config["seed"])
        calculate_metrics(
            model=TemperPLS(
                pls=pls,
                particles=particles,
                x_calibration=experiment_data.validation.x,
                y_calibration=experiment_data.validation.y,
            ),
            particles=particles,
            model_name=f"{pls_name}-temper",
            dataset_name=dataset_name,
            experiment_data=experiment_data,
            results_path=results_path,
            plots_path=plots_path,
            coverage=metrics_config["coverage"],
        )
        set_seed(pls_config["seed"])
        calculate_metrics(
            model=ConformalisePLS(
                pls=pls,
                particles=particles,
                x_calibration=experiment_data.validation.x,
                y_calibration=experiment_data.validation.y,
            ),
            particles=particles,
            model_name=f"{pls_name}-conformalise",
            dataset_name=dataset_name,
            experiment_data=experiment_data,
            results_path=results_path,
            plots_path=plots_path,
            coverage=metrics_config["coverage"],
        )

    student_likelihood = gpytorch.likelihoods.StudentTLikelihood()
    # hacky solution, should have this as a fixed value
    student_likelihood.register_constraint(
        param_name="raw_deg_free",
        constraint=gpytorch.constraints.Interval(
            lower_bound=degrees_of_freedom - 1e-10,
            upper_bound=degrees_of_freedom + 1e-10,
        ),
    )
    likelihood_dict = {
        "svgp": gpytorch.likelihoods.GaussianLikelihood(),
        "svgp-student": student_likelihood,
    }
    for model_name, likelihood in likelihood_dict.items():
        svgp_model_path = os.path.join(models_path, f"{model_name}.pth")
        if os.path.exists(svgp_model_path):
            svgp, _, _ = load_svgp(
                model_path=svgp_model_path,
                x_induce=inducing_points.x,
                mean=gpytorch.means.ConstantMean(),
                kernel=deepcopy(pls_kernel),
                likelihood=likelihood,
                learn_inducing_locations=False,
            )
        else:
            svgp, losses, best_learning_rate = train_svgp_runner(
                model_name=model_name,
                experiment_data=experiment_data,
                inducing_points=inducing_points,
                mean=gpytorch.means.ConstantMean(),
                kernel=deepcopy(pls_kernel),
                likelihood=likelihood,
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
                models_path=os.path.join(
                    models_path, f"{model_name}-kernel-iterations"
                ),
                plot_title=f"{dataset_name}",
                plot_loss_path=plots_path,
            )
            torch.save(
                {
                    "model": svgp.state_dict(),
                    "losses": losses,
                    "best_learning_rate": best_learning_rate,
                },
                os.path.join(models_path, f"{model_name}.pth"),
            )
        set_seed(svgp_config["seed"])
        calculate_metrics(
            model=svgp,
            model_name=model_name,
            dataset_name=dataset_name,
            experiment_data=experiment_data,
            results_path=results_path,
            plots_path=plots_path,
            coverage=metrics_config["coverage"],
        )
        set_seed(svgp_config["seed"])
        calculate_metrics(
            model=TemperGP(
                gp=svgp,
                x_calibration=experiment_data.validation.x,
                y_calibration=experiment_data.validation.y,
            ),
            model_name=f"{model_name}-temper",
            dataset_name=dataset_name,
            experiment_data=experiment_data,
            results_path=results_path,
            plots_path=plots_path,
            coverage=metrics_config["coverage"],
        )
        set_seed(svgp_config["seed"])
        calculate_metrics(
            model=ConformaliseGP(
                gp=svgp,
                x_calibration=experiment_data.validation.x,
                y_calibration=experiment_data.validation.y,
            ),
            experiment_data=experiment_data,
            model_name=f"{model_name}-conformalise",
            dataset_name=dataset_name,
            results_path=results_path,
            plots_path=plots_path,
            coverage=metrics_config["coverage"],
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    if args.data_seed == -1:
        data_seeds = list(range(10))
    else:
        data_seeds = [args.data_seed]

    outputs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
    for data_seed in data_seeds:
        for dataset_schema in RegressionDatasetSchema:
            try:
                main(
                    data_seed=data_seed,
                    dataset_name=str(dataset_schema.name),
                    data_config=loaded_config["data"],
                    kernel_config=loaded_config["kernel"],
                    inducing_points_config=loaded_config["inducing_points"],
                    pls_config=loaded_config["pls"],
                    svgp_config=loaded_config["svgp"],
                    metrics_config=loaded_config["metrics"],
                    outputs_path=outputs_path,
                )
            except Exception as e:
                print(f"Error with {dataset_schema.name=} and {data_seed=}:{e}")
            try:
                concatenate_metrics(
                    results_path=os.path.join(outputs_path, str(data_seed), "results"),
                    data_types=["train", "test"],
                    model_names=MODEL_NAMES,
                    datasets=list(RegressionDatasetSchema.__members__.keys()),
                    metrics=METRICS,
                )
            except Exception as e:
                print(f"Error with concatenating metrics for {data_seed=}:{e}")
            concatenate_metrics(
                results_path=os.path.join(outputs_path, str(data_seed), "results"),
                data_types=["train", "test"],
                model_names=MODEL_NAMES,
                datasets=list(RegressionDatasetSchema.__members__.keys()),
                metrics=METRICS,
            )
