import argparse
import math
import os
from copy import deepcopy
from typing import Any, Dict

import gpytorch
import pandas as pd
import torch
import yaml

from experiments.constructors import (
    construct_average_ard_kernel,
    construct_average_gaussian_likelihood,
)
from experiments.data import ExperimentData
from experiments.loaders import load_pls, load_svgp
from experiments.metrics import calculate_metrics, concatenate_metrics
from experiments.preprocess import set_up_experiment
from experiments.runners import (
    learn_subsample_gps,
    select_inducing_points,
    train_pls,
    train_svgp,
)
from experiments.uci.constants import DATASET_SCHEMA_MAPPING
from experiments.uci.schemas import DatasetSchema
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.kernels import PLSKernel
from src.projected_langevin_sampling import PLSRegressionIPB, PLSRegressionONB
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


def get_experiment_data(
    seed: int,
    train_data_percentage: float,
    dataset_name: str,
) -> ExperimentData:
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
        seed=seed,
        x=x,
        y=y,
        train_data_percentage=train_data_percentage,
        normalise=True,
    )
    return experiment_data


def main(
    data_seed: int,
    data_config: Dict[str, Any],
    kernel_config: Dict[str, Any],
    inducing_points_config: Dict[str, Any],
    pls_config: Dict[str, Any],
    svgp_config: Dict[str, Any],
) -> None:
    for dataset_schema in DatasetSchema:
        dataset_name = str(dataset_schema.name)
        print(f"Running experiment for {dataset_name=} and {data_seed=}.")

        data_path = f"experiments/uci/outputs/{data_seed}/data/{dataset_name}"
        plots_path = f"experiments/uci/outputs/{data_seed}/plots/{dataset_name}"
        results_path = f"experiments/uci/outputs/{data_seed}/results/{dataset_name}"
        models_path = f"experiments/uci/outputs/{data_seed}/models/{dataset_name}"
        subsample_gp_model_path = os.path.join(models_path, "subsample_gp")
        subsample_gp_data_path = os.path.join(data_path, "subsample_gp")
        os.makedirs(data_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)

        experiment_data_path = os.path.join(data_path, "experiment_data.pth")
        inducing_points_path = os.path.join(data_path, "inducing_points.pth")

        if os.path.exists(experiment_data_path):
            experiment_data = ExperimentData.load(experiment_data_path)
            print(f"Loaded experiment data from {experiment_data_path=}")
        else:
            experiment_data = get_experiment_data(
                seed=data_seed,
                train_data_percentage=data_config["train_data_percentage"],
                dataset_name=dataset_name,
            )
            experiment_data.save(experiment_data_path)

        subsample_gp_models = learn_subsample_gps(
            experiment_data=experiment_data,
            kernel=gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    ard_num_dims=experiment_data.train.x.shape[1]
                )
            ),
            subsample_size=kernel_config["subsample_size"],
            seed=kernel_config["seed"],
            number_of_epochs=kernel_config["number_of_epochs"],
            learning_rate=kernel_config["learning_rate"],
            number_of_iterations=kernel_config["number_of_iterations"],
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
            torch.save(inducing_points, inducing_points_path)
        likelihood = construct_average_gaussian_likelihood(
            likelihoods=[model.likelihood for model in subsample_gp_models]
        )

        pls_kernel = PLSKernel(
            base_kernel=average_ard_kernel,
            approximation_samples=inducing_points.x,
        )
        pls_dict = {
            "pls-onb": PLSRegressionONB(
                kernel=pls_kernel,
                x_induce=inducing_points.x,
                y_induce=inducing_points.y,
                x_train=experiment_data.train.x,
                y_train=experiment_data.train.y,
                jitter=pls_config["jitter"],
                observation_noise=float(likelihood.noise),
            ),
            "pls-ipb": PLSRegressionIPB(
                kernel=pls_kernel,
                x_induce=inducing_points.x,
                y_induce=inducing_points.y,
                x_train=experiment_data.train.x,
                y_train=experiment_data.train.y,
                jitter=pls_config["jitter"],
                observation_noise=float(likelihood.noise),
            ),
        }
        for pls_name, pls in pls_dict.items():
            pls_path = os.path.join(models_path, f"{pls_name}.pth")
            if os.path.exists(pls_path):
                pls, particles = load_pls(
                    pls=pls,
                    model_path=pls_path,
                )
            else:
                particles = train_pls(
                    pls=pls,
                    number_of_particles=pls_config["number_of_particles"],
                    particle_name=pls_name,
                    experiment_data=experiment_data,
                    inducing_points=inducing_points,
                    simulation_duration=pls_config["simulation_duration"],
                    maximum_number_of_steps=pls_config["maximum_number_of_steps"],
                    step_size_upper=pls_config["step_size_upper"],
                    number_of_step_searches=pls_config["number_of_step_searches"],
                    minimum_change_in_energy_potential=pls_config[
                        "minimum_change_in_energy_potential"
                    ],
                    seed=pls_config["seed"],
                    observation_noise_upper=pls_config["observation_noise_upper"],
                    observation_noise_lower=pls_config["observation_noise_lower"],
                    number_of_observation_noise_searches=pls_config[
                        "number_of_observation_noise_searches"
                    ],
                    plot_title=f"{dataset_name}",
                    plot_particles_path=None,
                    animate_1d_path=None,
                    plot_update_magnitude_path=plots_path,
                    metric_to_minimise=pls_config["metric_to_minimise"],
                    initial_particles_noise_only=pls_config[
                        "initial_particles_noise_only"
                    ],
                    early_stopper_patience=pls_config["early_stopper_patience"],
                )
                torch.save(
                    {
                        "particles": particles,
                        "observation_noise": pls.observation_noise,
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
            )
        for kernel_name, kernel in zip(["k", "r"], [average_ard_kernel, pls_kernel]):
            model_name = f"svgp-{kernel_name}"
            svgp_model_path = os.path.join(models_path, f"{model_name}.pth")
            if os.path.exists(svgp_model_path):
                svgp, _ = load_svgp(
                    model_path=svgp_model_path,
                    x_induce=inducing_points.x,
                    mean=gpytorch.means.ConstantMean(),
                    kernel=deepcopy(kernel),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(),
                    learn_inducing_locations=False,
                )
            else:
                svgp, losses = train_svgp(
                    model_name=model_name,
                    experiment_data=experiment_data,
                    inducing_points=inducing_points,
                    mean=gpytorch.means.ConstantMean(),
                    kernel=deepcopy(kernel),
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
                    models_path=os.path.join(
                        models_path, f"{model_name}-kernel-iterations"
                    ),
                    plot_title=f"{dataset_name}",
                    plot_1d_path=None,
                    animate_1d_path=None,
                    plot_loss_path=plots_path,
                )
                torch.save(
                    {
                        "model": svgp.state_dict(),
                        "losses": losses,
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
            )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    if args.data_seed == -1:
        data_seeds = [0, 1, 2, 3, 4]
    else:
        data_seeds = [args.data_seed]
    for data_seed in data_seeds:
        main(
            data_seed=data_seed,
            data_config=loaded_config["data"],
            kernel_config=loaded_config["kernel"],
            inducing_points_config=loaded_config["inducing_points"],
            pls_config=loaded_config["pls"],
            svgp_config=loaded_config["svgp"],
        )
        concatenate_metrics(
            results_path=f"experiments/uci/outputs/{data_seed}/results",
            data_types=["train", "test"],
            model_names=[
                "pls-onb",
                "pls-ipb",
                "svgp-k",
                "svgp-r",
            ],
            datasets=list(DatasetSchema.__members__.keys()),
            metrics=["mae", "mse", "nll"],
        )
