import argparse
import math
import os
from copy import deepcopy
from typing import Any, Dict

import gpytorch
import pandas as pd
import torch
import yaml

from experiments.constructors import construct_average_ard_kernel
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
from experiments.uci.constants import (
    DATASET_SCHEMA_MAPPING,
    ClassificationDatasetSchema,
)
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.kernels import PLSKernel
from src.projected_langevin_sampling import ProjectedLangevinSampling
from src.projected_langevin_sampling.basis import OrthonormalBasis
from src.projected_langevin_sampling.costs import BernoulliCost
from src.projected_langevin_sampling.link_functions import SigmoidLinkFunction
from src.utils import set_seed

parser = argparse.ArgumentParser(
    description="Main script for UCI classification data experiments."
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
    for column in input_column_names:
        if isinstance(df[column].dtypes, object) and len(df[column].unique()) == 2:
            df[column] = pd.get_dummies(df[column]).to_numpy()[:, 0].astype(float)
        df = df[pd.to_numeric(df[column], errors="coerce").notnull()]
        df[column] = pd.to_numeric(df[column]).astype(float)

    x = torch.tensor(df[input_column_names].to_numpy()).detach().double()
    y = (
        torch.tensor(pd.get_dummies(df[output_column_name]).to_numpy()[:, 0])
        .detach()
        .type(torch.int32)
    )

    experiment_data = set_up_experiment(
        name=dataset_name,
        problem_type=ProblemType.CLASSIFICATION,
        seed=seed,
        x=x,
        y=y,
        train_data_percentage=train_data_percentage,
        normalise=False,
    )
    return experiment_data


def main(
    data_seed: int,
    dataset_name: str,
    data_config: Dict[str, Any],
    kernel_config: Dict[str, Any],
    inducing_points_config: Dict[str, Any],
    pls_config: Dict[str, Any],
    svgp_config: Dict[str, Any],
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
            path=experiment_data_path, problem_type=ProblemType.CLASSIFICATION
        )
        print(f"Loaded experiment data from {experiment_data_path=}")
    else:
        experiment_data = get_experiment_data(
            seed=data_seed,
            train_data_percentage=data_config["train_data_percentage"],
            dataset_name=dataset_name,
        )
        experiment_data.save(experiment_data_path)
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
        model_path=subsample_gp_model_path,
        data_path=subsample_gp_data_path,
        plot_1d_subsample_path=None,
        plot_loss_path=plots_path,
        number_of_classes=likelihood.num_classes,
        early_stopper_patience=kernel_config["early_stopper_patience"],
    )
    experiment_data.train.y = y_train_labels
    average_ard_kernel = construct_average_ard_kernel(
        kernels=[model.kernel for model in subsample_gp_models],
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
    pls_kernel = PLSKernel(
        base_kernel=average_ard_kernel,
        approximation_samples=inducing_points.x,
    )
    onb_basis = OrthonormalBasis(
        kernel=pls_kernel,
        x_induce=inducing_points.x,
        x_train=experiment_data.train.x,
    )
    cost = BernoulliCost(
        y_train=experiment_data.train.y,
        link_function=SigmoidLinkFunction(),
    )
    pls_dict = {
        "pls-onb": ProjectedLangevinSampling(
            basis=onb_basis,
            cost=cost,
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
        )

    model_name = f"svgp"
    svgp_model_path = os.path.join(models_path, f"{model_name}.pth")
    if os.path.exists(svgp_model_path):
        svgp, _, _ = load_svgp(
            model_path=svgp_model_path,
            x_induce=inducing_points.x,
            mean=gpytorch.means.ConstantMean(),
            kernel=deepcopy(pls_kernel),
            likelihood=gpytorch.likelihoods.BernoulliLikelihood(),
            learn_inducing_locations=False,
        )
    else:
        svgp, losses, best_learning_rate = train_svgp_runner(
            model_name=model_name,
            experiment_data=experiment_data,
            inducing_points=inducing_points,
            mean=gpytorch.means.ConstantMean(),
            kernel=deepcopy(pls_kernel),
            likelihood=gpytorch.likelihoods.BernoulliLikelihood(),
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
            models_path=os.path.join(models_path, f"{model_name}-kernel-iterations"),
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
    outputs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
    for data_seed in data_seeds:
        for dataset_schema in ClassificationDatasetSchema:
            dataset_name = str(dataset_schema.name)
            main(
                data_seed=data_seed,
                dataset_name=dataset_name,
                data_config=loaded_config["data"],
                kernel_config=loaded_config["kernel"],
                inducing_points_config=loaded_config["inducing_points"],
                pls_config=loaded_config["pls"],
                svgp_config=loaded_config["svgp"],
                outputs_path=outputs_path,
            )
        concatenate_metrics(
            results_path=os.path.join(outputs_path, str(data_seed), "results"),
            data_types=["train", "test"],
            model_names=[
                "pls-onb",
                "svgp",
            ],
            datasets=list(ClassificationDatasetSchema.__members__.keys()),
            metrics=["mae", "mse", "nll", "acc", "auc", "f1"],
        )
