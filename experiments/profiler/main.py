import argparse
import os
from typing import Any, Dict, Optional, Type

import gpytorch
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from torch.profiler import ProfilerActivity, profile, record_function

from experiments.curves.curves import CURVE_FUNCTIONS
from experiments.data import Data, ExperimentData, ProblemType
from experiments.runners import select_inducing_points
from experiments.utils import create_directory
from src.gps import svGP
from src.inducing_point_selectors import ConditionalVarianceInducingPointSelector
from src.kernels.projected_langevin_sampling import PLSKernel
from src.projected_langevin_sampling import ProjectedLangevinSampling
from src.projected_langevin_sampling.basis import OrthonormalBasis
from src.projected_langevin_sampling.costs import GaussianCost
from src.projected_langevin_sampling.link_functions import IdentityLinkFunction

parser = argparse.ArgumentParser(
    description="Main script for profiling PLS and svGP models."
)
parser.add_argument("--config_path", type=str)


def get_experiment_data(
    number_of_data_points: int,
    seed: int,
    sigma_true: float = 0.2,
) -> ExperimentData:
    curve_function = CURVE_FUNCTIONS[0]
    x = torch.linspace(-3, 3, number_of_data_points).reshape(-1, 1)
    y = curve_function.regression(
        seed=seed,
        x=x,
        sigma_true=sigma_true,
    )
    experiment_data = ExperimentData(
        name=type(curve_function).__name__.lower(),
        problem_type=ProblemType.REGRESSION,
        full=Data(x=x, y=y, name="full"),
        train=Data(x=x, y=y, name="train"),
    )
    return experiment_data


def train_pls(
    pls_kernel: PLSKernel,
    inducing_points: Data,
    experiment_data: ExperimentData,
    observation_noise: float,
    number_of_particles: int,
    number_of_epochs: int,
    step_size: float,
) -> None:
    onb_basis = OrthonormalBasis(
        kernel=pls_kernel,
        x_induce=inducing_points.x,
        x_train=experiment_data.train.x,
    )
    cost = GaussianCost(
        observation_noise=float(observation_noise),
        y_train=experiment_data.train.y,
        link_function=IdentityLinkFunction(),
    )
    pls = ProjectedLangevinSampling(
        basis=onb_basis,
        cost=cost,
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
        loss = -mll(model(train_data.x), train_data.y)
        loss.backward()
        optimizer.step()


def parse_profiler(profiler: profile):
    df = pd.DataFrame({e.key: e.__dict__ for e in profiler.key_averages()}).T
    return pd.DataFrame(
        [
            [
                df.loc["model_training", "cpu_time_total"]
                / 1e3
                * df.loc["model_training", "count"]
            ]
        ],
        columns=["cpu_time_milliseconds"],
    )


def profile_pls(
    pls_kernel: PLSKernel,
    observation_noise: float,
    inducing_points: Data,
    experiment_data: ExperimentData,
    number_of_particles: int,
    number_of_epochs: int,
    df_path: str,
    step_size: float = 1e-10,
) -> pd.DataFrame:
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_training"):
            train_pls(
                pls_kernel=pls_kernel,
                inducing_points=inducing_points,
                experiment_data=experiment_data,
                observation_noise=observation_noise,
                number_of_particles=number_of_particles,
                number_of_epochs=number_of_epochs,
                step_size=step_size,
            )
    df = parse_profiler(
        profiler=prof,
    )
    df.to_csv(
        df_path,
        index=False,
    )
    return df


def profile_svgp(
    inducing_points: Data,
    experiment_data: ExperimentData,
    kernel: gpytorch.kernels.Kernel,
    number_of_epochs: int,
    df_path: str,
    learning_rate: float = 1e-10,
) -> pd.DataFrame:
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_training"):
            train_svgp(
                number_of_epochs=number_of_epochs,
                kernel=kernel,
                train_data=experiment_data.train,
                inducing_points=inducing_points,
                learning_rate=learning_rate,
            )
    df = parse_profiler(
        profiler=prof,
    )
    df.to_csv(
        df_path,
        index=False,
    )
    return df


def run_experiment(
    seed: int,
    number_of_data_points: int,
    number_induce_points: int,
    number_of_epochs: int,
    number_of_particles: int,
    models_path: str,
    data_path: str,
    results_path: str,
    observation_noise: float = 0.01,
):
    data_dir = os.path.join(data_path, f"seed_{seed}", f"n_{number_of_data_points}")
    create_directory(data_dir)
    model_dir = os.path.join(models_path, f"seed_{seed}", f"n_{number_of_data_points}")
    create_directory(model_dir)

    experiment_data_path = os.path.join(data_dir, "experiment_data.pth")
    if os.path.exists(experiment_data_path):
        experiment_data = ExperimentData.load(
            path=experiment_data_path, problem_type=ProblemType.CLASSIFICATION
        )
        print(f"Loaded experiment data from {experiment_data_path=}")
    else:
        experiment_data = get_experiment_data(
            number_of_data_points=number_of_data_points,
            seed=seed,
        )
        experiment_data.save(experiment_data_path)
    ard_kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=experiment_data.train.x.shape[1])
    )

    induce_data_dir = os.path.join(data_dir, f"m_{number_induce_points}")
    create_directory(induce_data_dir)
    inducing_points_path = os.path.join(induce_data_dir, "inducing_points.pth")
    if os.path.exists(inducing_points_path):
        inducing_points = torch.load(inducing_points_path)
        print(f"Loaded inducing points from {inducing_points_path=}")
    else:
        inducing_points = select_inducing_points(
            seed=seed,
            inducing_point_selector=ConditionalVarianceInducingPointSelector(),
            data=experiment_data.train,
            number_induce_points=number_induce_points,
            kernel=ard_kernel,
        )
        torch.save(inducing_points, inducing_points_path)
    pls_kernel = PLSKernel(
        base_kernel=ard_kernel,
        approximation_samples=inducing_points.x,
    )
    df_dir = os.path.join(
        results_path,
        f"seed_{seed}",
        f"n_{number_of_data_points}",
        f"m_{number_induce_points}",
        f"t_{number_of_epochs}",
        "pls",
        f"j_{number_of_particles}",
    )
    create_directory(df_dir)
    df_path = os.path.join(df_dir, "profile.csv")
    if os.path.exists(df_path):
        df_pls = pd.read_csv(df_path)
        print(f"Loaded profile from {df_path=}")
    else:
        df_pls = profile_pls(
            pls_kernel=pls_kernel,
            observation_noise=observation_noise,
            inducing_points=inducing_points,
            experiment_data=experiment_data,
            number_of_particles=number_of_particles,
            number_of_epochs=number_of_epochs,
            df_path=df_path,
        )
    df_pls["model"] = "PLS"
    df_pls["seed"] = seed
    df_pls["n"] = number_of_data_points
    df_pls["m"] = number_induce_points
    df_pls["t"] = number_of_epochs
    df_pls["j"] = number_of_particles

    df_dir = os.path.join(
        results_path,
        f"seed_{seed}",
        f"n_{number_of_data_points}",
        f"m_{number_induce_points}",
        f"t_{number_of_epochs}",
        "svgp",
    )
    create_directory(df_dir)
    df_path = os.path.join(df_dir, "profile.csv")
    if os.path.exists(df_path):
        df_svgp = pd.read_csv(df_path)
        print(f"Loaded profile from {df_path=}")
    else:
        df_svgp = profile_svgp(
            inducing_points=inducing_points,
            experiment_data=experiment_data,
            kernel=pls_kernel,
            number_of_epochs=number_of_epochs,
            df_path=df_path,
        )
    df_svgp["model"] = "SVGP"
    df_svgp["seed"] = seed
    df_svgp["n"] = number_of_data_points
    df_svgp["m"] = number_induce_points
    df_svgp["t"] = number_of_epochs
    return df_pls, df_svgp


def plot_df(
    x_axis: str,
    x_axis_name: str,
    save_path: str,
    df_pls: Optional[pd.DataFrame] = None,
    df_svgp: Optional[pd.DataFrame] = None,
):
    def _plot_df(fig, ax, df):
        df_mean = (
            df[["model", x_axis, "cpu_time_milliseconds"]]
            .groupby([x_axis, "model"])
            .mean()
            .reset_index()
        )
        df_std = (
            df[["model", x_axis, "cpu_time_milliseconds"]]
            .groupby([x_axis, "model"])
            .std()
            .reset_index()
        )
        for model in df["model"].unique():
            df_mean = df_mean[df_mean["model"] == model].sort_values(x_axis)
            df_std = df_std[df_std["model"] == model].sort_values(x_axis)
            ax.errorbar(
                df_mean[x_axis],
                df_mean["cpu_time_milliseconds"],
                yerr=2 * df_std["cpu_time_milliseconds"],
                label=f"{model} (±2 stdev)",
                capsize=3,
                marker=".",
                markersize=10,
            )
        return fig, ax

    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    if df_pls is not None:
        fig, ax = _plot_df(fig, ax, df_pls)
    if df_svgp is not None:
        fig, ax = _plot_df(fig, ax, df_svgp)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel("CPU Time (milliseconds)")
    ax.set_title(f"CPU Time vs {x_axis_name}")
    fig.legend(
        loc="outside lower center",
        ncols=3,
    )
    fig.savefig(
        save_path,
    )
    plt.close(fig)


def main(
    number_of_data_points_config: Dict[str, Any],
    number_of_induce_points_config: Dict[str, Any],
    number_of_epochs_config: Dict[str, Any],
    number_of_particles_config: Dict[str, Any],
    profiler_config: Dict[str, Any],
) -> None:
    profiler_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
    models_path = f"{profiler_path}/models"
    data_path = f"{profiler_path}/data"
    results_path = f"{profiler_path}/results"
    create_directory(models_path)
    create_directory(data_path)
    create_directory(results_path)
    df_pls_list = []
    df_svgp_list = []
    for seed in range(profiler_config["number_of_seeds"]):
        for number_of_data_points in range(
            number_of_data_points_config["start"],
            number_of_data_points_config["stop"] + 1,
            number_of_data_points_config["step"],
        ):
            df_pls, df_svgp = run_experiment(
                seed=seed,
                number_of_data_points=number_of_data_points,
                number_induce_points=number_of_induce_points_config["default"],
                number_of_epochs=number_of_epochs_config["default"],
                number_of_particles=number_of_particles_config["default"],
                models_path=models_path,
                data_path=data_path,
                results_path=results_path,
            )
            df_pls_list.append(df_pls)
            df_svgp_list.append(df_svgp)
        for number_induce_points in range(
            number_of_induce_points_config["start"],
            number_of_induce_points_config["stop"] + 1,
            number_of_induce_points_config["step"],
        ):
            df_pls, df_svgp = run_experiment(
                seed=seed,
                number_of_data_points=number_of_data_points_config["default"],
                number_induce_points=number_induce_points,
                number_of_epochs=number_of_epochs_config["default"],
                number_of_particles=number_of_particles_config["default"],
                models_path=models_path,
                data_path=data_path,
                results_path=results_path,
            )
            df_pls_list.append(df_pls)
            df_svgp_list.append(df_svgp)
        for number_of_epochs in range(
            number_of_epochs_config["start"],
            number_of_epochs_config["stop"] + 1,
            number_of_epochs_config["step"],
        ):
            df_pls, df_svgp = run_experiment(
                seed=seed,
                number_of_data_points=number_of_data_points_config["default"],
                number_induce_points=number_of_induce_points_config["default"],
                number_of_epochs=number_of_epochs,
                number_of_particles=number_of_particles_config["default"],
                models_path=models_path,
                data_path=data_path,
                results_path=results_path,
            )
            df_pls_list.append(df_pls)
            df_svgp_list.append(df_svgp)
        for number_of_particles in range(
            number_of_particles_config["start"],
            number_of_particles_config["stop"] + 1,
            number_of_particles_config["step"],
        ):
            df_pls, df_svgp = run_experiment(
                seed=seed,
                number_of_data_points=number_of_data_points_config["default"],
                number_induce_points=number_of_induce_points_config["default"],
                number_of_epochs=number_of_epochs_config["default"],
                number_of_particles=number_of_particles,
                models_path=models_path,
                data_path=data_path,
                results_path=results_path,
            )
            df_pls_list.append(df_pls)
            df_svgp_list.append(df_svgp)
    df_pls = pd.concat(df_pls_list).drop_duplicates()
    df_svgp = pd.concat(df_svgp_list).drop_duplicates()
    plot_df(
        df_pls=df_pls[
            (df_pls["m"] == number_of_induce_points_config["default"])
            & (df_pls["t"] == number_of_epochs_config["default"])
            & (df_pls["j"] == number_of_particles_config["default"])
        ],
        df_svgp=df_svgp[
            (df_svgp["m"] == number_of_induce_points_config["default"])
            & (df_svgp["t"] == number_of_epochs_config["default"])
        ],
        x_axis="n",
        x_axis_name="Number of Training Points (N)",
        save_path=f"{profiler_path}/number_of_training_points.png",
    )
    plot_df(
        df_pls=df_pls[
            (df_pls["n"] == number_of_data_points_config["default"])
            & (df_pls["t"] == number_of_epochs_config["default"])
            & (df_pls["j"] == number_of_particles_config["default"])
        ],
        df_svgp=df_svgp[
            (df_svgp["n"] == number_of_data_points_config["default"])
            & (df_svgp["t"] == number_of_epochs_config["default"])
        ],
        x_axis="m",
        x_axis_name="Number of Inducing Points (M)",
        save_path=f"{profiler_path}/number_of_inducing_points.png",
    )
    plot_df(
        df_pls=df_pls[
            (df_pls["n"] == number_of_data_points_config["default"])
            & (df_pls["m"] == number_of_induce_points_config["default"])
            & (df_pls["j"] == number_of_particles_config["default"])
        ],
        df_svgp=df_svgp[
            (df_svgp["n"] == number_of_data_points_config["default"])
            & (df_svgp["m"] == number_of_induce_points_config["default"])
        ],
        x_axis="t",
        x_axis_name="Number of Epochs (T)",
        save_path=f"{profiler_path}/number_of_epochs.png",
    )
    plot_df(
        df_pls=df_pls[
            (df_pls["n"] == number_of_data_points_config["default"])
            & (df_pls["m"] == number_of_induce_points_config["default"])
            & (df_pls["t"] == number_of_epochs_config["default"])
        ],
        x_axis="j",
        x_axis_name="Number of Particles (J)",
        save_path=f"{profiler_path}/number_of_particles.png",
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    main(
        number_of_data_points_config=loaded_config["number_of_data_points"],
        number_of_induce_points_config=loaded_config["number_of_induce_points"],
        number_of_epochs_config=loaded_config["number_of_epochs"],
        number_of_particles_config=loaded_config["number_of_particles"],
        profiler_config=loaded_config["profiler"],
    )
