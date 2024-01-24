import argparse
import math
import os
from typing import Any, Dict, Type, Union

import gpytorch
import numpy as np
import pandas as pd
import torch
import yaml
from torch.profiler import ProfilerActivity, profile, record_function

from experiments.constructors import (
    construct_average_ard_kernel,
    construct_average_gaussian_likelihood,
)
from experiments.curves.curves import CURVE_FUNCTIONS, Curve
from experiments.data import Data, ExperimentData, ProblemType
from experiments.preprocess import split_regression_data_intervals
from experiments.runners import learn_subsample_gps, select_inducing_points
from experiments.utils import create_directory
from src.gps import svGP
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.kernels.projected_langevin_sampling import PLSKernel
from src.projected_langevin_sampling import PLSRegressionIPB, PLSRegressionONB

parser = argparse.ArgumentParser(
    description="Main script for toy regression curves experiments."
)
parser.add_argument("--config_path", type=str)


def get_experiment_data(
    curve_function: Curve,
    number_of_data_points: int,
    seed: int,
    sigma_true: float,
    number_of_test_intervals: int,
    total_number_of_intervals: int,
) -> ExperimentData:
    x = torch.linspace(-3, 3, number_of_data_points).reshape(-1, 1)
    y = curve_function.regression(
        seed=seed,
        x=x,
        sigma_true=sigma_true,
    )
    (
        x_train,
        y_train,
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
    experiment_data = ExperimentData(
        name=type(curve_function).__name__.lower(),
        problem_type=ProblemType.REGRESSION,
        full=Data(x=x, y=y, name="full"),
        train=Data(x=x_train, y=y_train, name="train"),
        test=Data(x=x_test, y=y_test, name="test"),
    )
    return experiment_data


def train_pls(
    pls_object: Type[Union["PLSRegressionONB", "PLSRegressionIPB"]],
    pls_kernel: PLSKernel,
    inducing_points: Data,
    experiment_data: ExperimentData,
    observation_noise: float,
    number_of_particles: int,
    number_of_epochs: int,
    step_size: float,
) -> None:
    pls = pls_object(
        kernel=pls_kernel,
        x_induce=inducing_points.x,
        y_induce=inducing_points.y,
        x_train=experiment_data.train.x,
        y_train=experiment_data.train.y,
        observation_noise=observation_noise,
    )
    particles = pls.initialise_particles(
        number_of_particles=number_of_particles,
        noise_only=True,
    )
    for _ in range(number_of_epochs):
        particle_update = pls.calculate_particle_update(
            particles=particles,
            step_size=step_size,
        )
        particles += particle_update


def parse_profiler(profiler: profile):
    return pd.DataFrame({e.key: e.__dict__ for e in profiler.key_averages()}).T


def train_svgp(
    train_data: Data,
    inducing_points: Data,
    kernel: gpytorch.kernels.Kernel,
    number_of_epochs: int,
    learning_rate: float,
) -> None:
    model = svGP(
        mean=gpytorch.means.ConstantMean(),
        kernel=kernel,
        x_induce=inducing_points.x,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        learn_inducing_locations=False,
    )
    all_params = set(model.parameters())
    if isinstance(model.kernel, PLSKernel):
        all_params -= {model.kernel.base_kernel.base_kernel.raw_lengthscale}
        all_params -= {model.kernel.base_kernel.raw_outputscale}
    else:
        all_params -= {model.kernel.base_kernel.raw_lengthscale}
        all_params -= {model.kernel.raw_outputscale}
    model.likelihood.noise_covar.noise.data.fill_(1.0)
    model.likelihood.noise_covar.noise.data.requires_grad = False
    model.train()
    model.likelihood.train()
    optimizer = torch.optim.SGD(
        [
            {"params": list(all_params)},
        ],
        lr=learning_rate,
    )
    mll = gpytorch.mlls.VariationalELBO(
        model.likelihood, model, num_data=train_data.x.shape[0]
    )
    for _ in range(number_of_epochs):
        optimizer.zero_grad()
        output = model(train_data.x)
        loss = -mll(output, train_data.y)
        loss.backward()
        optimizer.step()


def main(
    curve_function: Curve,
    data_config: Dict[str, Any],
    kernel_config: Dict[str, Any],
    inducing_points_config: Dict[str, Any],
    pls_config: Dict[str, Any],
    svgp_config: Dict[str, Any],
    profiler_config: Dict[str, Any],
) -> None:
    experiment_data = get_experiment_data(
        curve_function=curve_function,
        number_of_data_points=data_config["number_of_data_points"],
        seed=data_config["seed"],
        sigma_true=data_config["sigma_true"],
        number_of_test_intervals=data_config["number_of_test_intervals"],
        total_number_of_intervals=data_config["total_number_of_intervals"],
    )
    models_path = f"experiments/curves/profiler/outputs/models/{type(curve_function).__name__.lower()}"
    data_path = f"experiments/curves/profiler/outputs/data/{type(curve_function).__name__.lower()}"
    profiler_path = f"experiments/curves/profiler/outputs"
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
        plot_loss_path=None,
    )
    average_ard_kernel = construct_average_ard_kernel(
        kernels=[model.kernel for model in subsample_gp_models]
    )
    likelihood = construct_average_gaussian_likelihood(
        likelihoods=[model.likelihood for model in subsample_gp_models]
    )
    for number_induce_points in np.arange(10, 110, 10).astype(int):
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
        df_list = []
        for pls_name, pls in zip(
            ["pls-onb", "pls-ipb"], [PLSRegressionONB, PLSRegressionIPB]
        ):
            df_model_list = []
            model_profiler_path = os.path.join(profiler_path, pls_name)
            create_directory(model_profiler_path)
            for i in range(profiler_config["number_of_repeats"]):
                with profile(
                    activities=[ProfilerActivity.CPU], record_shapes=True
                ) as prof:
                    with record_function("model_training"):
                        train_pls(
                            pls_object=pls,
                            pls_kernel=pls_kernel,
                            inducing_points=inducing_points,
                            experiment_data=experiment_data,
                            observation_noise=float(likelihood.noise),
                            number_of_particles=pls_config["number_of_particles"],
                            number_of_epochs=profiler_config["number_of_epochs"],
                            step_size=pls_config["step_size"],
                        )
                df = parse_profiler(
                    profiler=prof,
                )
                df.to_csv(
                    os.path.join(
                        model_profiler_path, f"{number_induce_points}-{i}.csv"
                    ),
                    index=False,
                )
                df_model_list.append(
                    pd.DataFrame(
                        [
                            [
                                df.loc["model_training", "cpu_time_total"]
                                / 1e6
                                * df.loc["model_training", "count"]
                            ]
                        ],
                        columns=[pls_name],
                    )
                )
            df_list.append(pd.concat(df_model_list, axis=0))

        for kernel_name, kernel in zip(["k", "r"], [average_ard_kernel, pls_kernel]):
            model_name = f"svgp-{kernel_name}"
            model_profiler_path = os.path.join(profiler_path, model_name)
            create_directory(model_profiler_path)
            df_model_list = []
            for i in range(profiler_config["number_of_repeats"]):
                with profile(
                    activities=[ProfilerActivity.CPU], record_shapes=True
                ) as prof:
                    with record_function("model_training"):
                        train_svgp(
                            number_of_epochs=profiler_config["number_of_epochs"],
                            kernel=kernel,
                            train_data=experiment_data.train,
                            inducing_points=inducing_points,
                            learning_rate=svgp_config["learning_rate"],
                        )
                df = parse_profiler(
                    profiler=prof,
                )
                df.to_csv(
                    os.path.join(
                        model_profiler_path, f"{number_induce_points}-{i}.csv"
                    ),
                    index=False,
                )
                df_model_list.append(
                    pd.DataFrame(
                        [
                            [
                                df.loc["model_training", "cpu_time_total"]
                                / 1e6
                                * df.loc["model_training", "count"]
                            ]
                        ],
                        columns=[model_name],
                    )
                )
            df_list.append(pd.concat(df_model_list, axis=0))
        pd.concat(df_list, axis=1).to_csv(
            os.path.join(profiler_path, f"total-{number_induce_points}.csv"),
            index=False,
        )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    for curve_function_ in CURVE_FUNCTIONS[:1]:
        main(
            curve_function=curve_function_,
            data_config=loaded_config["data"],
            kernel_config=loaded_config["kernel"],
            inducing_points_config=loaded_config["inducing_points"],
            pls_config=loaded_config["pls"],
            svgp_config=loaded_config["svgp"],
            profiler_config=loaded_config["profiler"],
        )
