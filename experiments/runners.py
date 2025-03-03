import math
import os
from copy import deepcopy
from typing import List, Tuple

import gpytorch
import numpy as np
import sklearn
import torch
from sklearn.neighbors import NearestNeighbors

from experiments.data import Data, ExperimentData
from experiments.loaders import load_ard_exact_gp_model, load_svgp
from experiments.metrics import calculate_mae, calculate_mse, calculate_nll
from experiments.plotters import (
    animate_1d_pls_predictions,
    animate_1d_pls_untransformed_predictions,
    plot_1d_gp_prediction_and_inducing_points,
    plot_1d_pls_prediction,
    plot_1d_pls_prediction_histogram,
    plot_energy_potentials,
    plot_losses,
)
from experiments.trainers import train_exact_gp, train_pls, train_svgp
from experiments.utils import create_directory
from src.conformalise import ConformalisePLS
from src.custom_types import PLS_TYPE
from src.gaussian_process import SVGP
from src.gaussian_process.exact_gp import ExactGP
from src.inducing_point_selectors import InducingPointSelector
from src.projected_langevin_sampling import PLS
from src.samplers import sample_point
from src.temper.pls import TemperPLS
from src.utils import set_seed


def inducing_points_runner(
    seed: int,
    inducing_point_selector: InducingPointSelector,
    data: Data,
    number_induce_points: int,
    kernel: gpytorch.kernels.Kernel,
) -> Data:
    set_seed(seed)
    x_induce, induce_indices = inducing_point_selector(
        x=torch.atleast_2d(data.x).reshape(data.x.shape[0], -1),
        m=number_induce_points,
        kernel=kernel,
    )
    y_induce = data.y[induce_indices]
    y_induce_untransformed = (
        data.y_untransformed[induce_indices]
        if data.y_untransformed is not None
        else None
    )
    return Data(
        x=x_induce,
        y=y_induce.type(data.y.dtype),
        y_untransformed=y_induce_untransformed.type(data.y_untransformed.dtype)
        if y_induce_untransformed is not None
        else None,
        name="induce",
    )


def load_subsample_data(
    data: Data,
    subsample_size: int,
):
    if subsample_size > len(data.x):
        return data
    knn = NearestNeighbors(
        n_neighbors=subsample_size,
        p=2,
    )
    knn.fit(X=data.x.cpu(), y=data.y.cpu())
    x_sample = sample_point(x=data.x)
    subsample_indices = knn.kneighbors(
        X=x_sample.cpu(),
        return_distance=False,
    ).flatten()
    return Data(
        x=data.x[subsample_indices],
        y=data.y[..., subsample_indices],
    )


def exact_gp_runner(
    experiment_data: ExperimentData,
    kernel: gpytorch.kernels.Kernel,
    likelihood: gpytorch.likelihoods.Likelihood,
    subsample_size: int,
    seed: int,
    number_of_epochs: int,
    learning_rate: float,
    number_of_iterations: int,
    early_stopper_patience: float,
    model_path: str,
    data_path: str,
    plot_1d_subsample_path: str | None = None,
    plot_loss_path: str | None = None,
    number_of_classes: int = 1,
) -> List[ExactGP]:
    """
    Trains an exact GP on the full data or a subsample of the data, depending on the training data size.
    """
    create_directory(model_path)
    create_directory(data_path)
    models = []
    losses_history = {}
    if subsample_size >= len(experiment_data.train.x):
        number_of_iterations = 1
        model_name = "full_exact_gp"
    else:
        model_name = "subsample_exact_gp"
    for i in range(number_of_iterations):
        subsample_model_path = os.path.join(
            model_path, f"{model_name}_{i+1}_of_{number_of_iterations}.pth"
        )
        subsample_data_path = os.path.join(
            data_path, f"{model_name}_{i+1}_of_{number_of_iterations}.pth"
        )
        set_seed(seed + i)
        mean = (
            gpytorch.means.ConstantMean(batch_shape=torch.Size((number_of_classes,)))
            if number_of_classes > 1
            else gpytorch.means.ConstantMean()
        )
        if os.path.exists(subsample_model_path) and os.path.exists(subsample_data_path):
            model, losses = load_ard_exact_gp_model(
                model_path=subsample_model_path,
                data_path=subsample_data_path,
                likelihood=likelihood,
                mean=mean,
                kernel=deepcopy(kernel),
            )
        else:
            data = load_subsample_data(
                data=experiment_data.train,
                subsample_size=subsample_size,
            )
            model, losses = train_exact_gp(
                data=data,
                mean=mean,
                kernel=deepcopy(kernel),
                seed=seed,
                number_of_epochs=number_of_epochs,
                learning_rate=learning_rate,
                likelihood=likelihood,
                early_stopper_patience=early_stopper_patience,
                model_name=model_name,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "losses": losses,
                },
                subsample_model_path,
            )
            torch.save(
                data,
                subsample_data_path,
            )
            if plot_1d_subsample_path is not None:
                create_directory(plot_1d_subsample_path)
                plot_1d_gp_prediction_and_inducing_points(
                    model=model,
                    experiment_data=experiment_data,
                    title=f"Subsample GP (iteration {i + 1}, {subsample_size=})",
                    save_path=os.path.join(
                        plot_1d_subsample_path,
                        f"gp-subsample-iteration-{i + 1}.png",
                    ),
                )
        losses_history[learning_rate] = losses
        models.append(model)
    if plot_loss_path is not None:
        create_directory(plot_loss_path)
        plot_losses(
            losses_history=losses_history,
            title=f"Subsample GP Learning ({subsample_size=})",
            save_path=os.path.join(
                plot_loss_path,
                "subsample-gp-losses.png",
            ),
        )
    return models


def plot_pls_1d_particles_runner(
    pls: PLS_TYPE,
    particles: torch.Tensor,
    particle_name: str,
    experiment_data: ExperimentData,
    plot_particles_path: str,
    inducing_points: Data | None = None,
    coverage: float = 0.95,
    plot_title: str | None = None,
    number_of_particles_to_plot: int | None = None,
) -> None:
    create_directory(plot_particles_path)
    if isinstance(pls, PLS):
        predicted_distribution = pls.predict(
            x=experiment_data.full.x,
            particles=particles,
        )
    elif isinstance(pls, ConformalisePLS):
        predicted_distribution = pls.predict(
            x=experiment_data.full.x,
            coverage=coverage,
        )
    elif isinstance(pls, TemperPLS):
        predicted_distribution = pls.predict(
            x=experiment_data.full.x,
        )
    else:
        raise TypeError(f"Unknown PLS type: {type(pls)}")
    predicted_samples = None
    if isinstance(pls, PLS):
        predicted_samples = pls.predict_samples(
            x=experiment_data.full.x,
            particles=particles[:, :number_of_particles_to_plot]
            if number_of_particles_to_plot is not None
            else particles,
        ).detach()
    plot_1d_pls_prediction(
        experiment_data=experiment_data,
        inducing_points=inducing_points,
        x=experiment_data.full.x,
        predicted_samples=predicted_samples,
        predicted_distribution=predicted_distribution,
        coverage=coverage,
        title=f"{plot_title}" if plot_title is not None else None,
        save_path=os.path.join(
            plot_particles_path,
            f"particles-{particle_name}.png",
        ),
    )
    if experiment_data.full.y_untransformed is not None:
        untransformed_predicted_samples = pls.predict_untransformed_samples(
            x=experiment_data.full.x,
            particles=particles[:, :number_of_particles_to_plot]
            if number_of_particles_to_plot is not None
            else particles,
        ).detach()
        plot_1d_pls_prediction(
            experiment_data=experiment_data,
            x=experiment_data.full.x,
            predicted_samples=untransformed_predicted_samples,
            y_name="$f(x)$",
            title=f"{plot_title}: $f(x)$" if plot_title is not None else None,
            save_path=os.path.join(
                plot_particles_path,
                f"untransformed-particles-{particle_name}.png",
            ),
            is_sample_untransformed=True,
        )
        if predicted_samples is not None:
            plot_1d_pls_prediction_histogram(
                experiment_data=experiment_data,
                x=experiment_data.full.x,
                predicted_samples=predicted_samples,
                untransformed_predicted_samples=untransformed_predicted_samples,
                title=f"{plot_title}" if plot_title is not None else None,
                save_path=os.path.join(
                    plot_particles_path,
                    f"particles-histogram-{particle_name}.png",
                ),
            )


def animate_pls_1d_particles_runner(
    pls: PLS,
    number_of_particles: int,
    particle_name: str,
    experiment_data: ExperimentData,
    seed: int,
    best_lr: float,
    number_of_epochs: int,
    plot_title: str | None = None,
    animate_1d_path: str | None = None,
    animate_1d_untransformed_path: str | None = None,
    christmas_colours: bool = False,
    initial_particles_noise_only: bool = False,
    init_particles: torch.Tensor | None = None,
):
    if best_lr is None:
        return
    if animate_1d_path is not None:
        animate_1d_pls_predictions(
            pls=pls,
            seed=seed,
            number_of_particles=number_of_particles,
            initial_particles_noise_only=initial_particles_noise_only,
            step_size=best_lr,
            number_of_epochs=number_of_epochs,
            experiment_data=experiment_data,
            x=experiment_data.full.x,
            title=plot_title,
            save_path=os.path.join(
                animate_1d_path,
                f"{particle_name}.gif",
            ),
            christmas_colours=christmas_colours,
            init_particles=init_particles.clone()
            if init_particles is not None
            else None,
        )
    if animate_1d_untransformed_path is not None:
        animate_1d_pls_untransformed_predictions(
            pls=pls,
            seed=seed,
            number_of_particles=number_of_particles,
            initial_particles_noise_only=initial_particles_noise_only,
            step_size=best_lr,
            number_of_epochs=number_of_epochs,
            experiment_data=experiment_data,
            x=experiment_data.full.x,
            title=plot_title,
            save_path=os.path.join(
                animate_1d_untransformed_path,
                f"untransformed-{particle_name}.gif",
            ),
            christmas_colours=christmas_colours,
            init_particles=init_particles.clone()
            if init_particles is not None
            else None,
        )


def train_pls_runner(
    pls: PLS,
    particle_name: str,
    experiment_data: ExperimentData,
    simulation_duration: float,
    maximum_number_of_steps: int,
    early_stopper_patience: float,
    number_of_step_searches: int,
    step_size_upper: float,
    minimum_change_in_energy_potential: float,
    seed: int,
    particles: torch.Tensor,
    plot_title: str | None = None,
    plot_energy_potential_path: str | None = None,
    metric_to_optimise: str = "nll",
) -> Tuple[torch.Tensor, float, int]:
    if metric_to_optimise in ["nll", "mse", "mae", "loss"]:
        best_metric_value = float("inf")
    elif metric_to_optimise in ["acc", "auc", "f1"]:
        best_metric_value = 0
    else:
        raise NotImplementedError(f"Unknown metric to optimise {metric_to_optimise}.")
    metric_value = None
    best_lr = None
    energy_potentials_history = {}
    step_sizes = np.logspace(
        np.log10(step_size_upper),
        np.log10(simulation_duration / maximum_number_of_steps),
        number_of_step_searches,
    )
    particles_out = particles.detach().clone()
    for i, step_size in enumerate(step_sizes):
        number_of_epochs = int(simulation_duration / step_size)
        set_seed(seed)
        particles_i, energy_potentials = train_pls(
            pls=pls,
            particles=particles.detach().clone(),
            number_of_epochs=number_of_epochs,
            step_size=step_size,
            early_stopper_patience=early_stopper_patience,
            tqdm_desc=f"PLS Step Size Search {i+1} of {number_of_step_searches} for {particle_name} ({step_size=})",
        )
        if energy_potentials and torch.isfinite(particles_i).all():
            energy_potentials_history[step_size] = energy_potentials
            prediction = pls.predict(
                x=experiment_data.train.x,
                particles=particles_i,
            )
            if metric_to_optimise == "nll":
                metric_value = calculate_nll(
                    prediction=prediction,
                    y=experiment_data.train.y,
                )
            elif metric_to_optimise == "mse":
                metric_value = calculate_mse(
                    prediction=prediction,
                    y=experiment_data.train.y,
                )
            elif metric_to_optimise == "mae":
                metric_value = calculate_mae(
                    prediction=prediction,
                    y=experiment_data.train.y,
                )
            elif metric_to_optimise == "acc":
                metric_value = sklearn.metrics.accuracy_score(
                    y_true=experiment_data.train.y.cpu().detach().numpy(),
                    y_pred=prediction.probs.round().cpu().detach().numpy(),
                )
            elif metric_to_optimise == "auc":
                metric_value = sklearn.metrics.roc_auc_score(
                    y_true=experiment_data.train.y.cpu().detach().numpy(),
                    y_score=prediction.probs.cpu().detach().numpy(),
                )
            elif metric_to_optimise == "f1":
                metric_value = sklearn.metrics.f1_score(
                    y_true=experiment_data.train.y.cpu().detach().numpy(),
                    y_pred=prediction.probs.round().cpu().detach().numpy(),
                )
            elif metric_to_optimise == "loss":
                metric_value = energy_potentials[-1]
            else:
                raise ValueError(f"Unknown metric to minimise: {metric_to_optimise}")
            if (
                metric_to_optimise in ["nll", "mse", "mae", "loss"]
                and metric_value < best_metric_value
            ) or (
                metric_to_optimise in ["acc", "auc", "f1"]
                and metric_value > best_metric_value
            ):
                best_metric_value = metric_value
                best_lr = step_size
                particles_out = deepcopy(particles_i.detach())
            if (
                i > 0
                and step_sizes[i - 1] in energy_potentials_history
                and abs(
                    energy_potentials_history[step_sizes[i - 1]][-1]
                    - energy_potentials[-1]
                )
                / energy_potentials_history[step_sizes[i - 1]][-1]
                < minimum_change_in_energy_potential
            ):
                break
    if energy_potentials_history and plot_energy_potential_path is not None:
        create_directory(plot_energy_potential_path)
        plot_energy_potentials(
            energy_potentials_history=energy_potentials_history,
            title=f"{plot_title} (energy potentials)"
            if plot_title is not None
            else None,
            save_path=os.path.join(
                plot_energy_potential_path,
                f"energy-potential-{particle_name}.png",
            ),
        )
    return particles_out, best_lr, len(energy_potentials_history[best_lr])


def train_svgp_runner(
    model_name: str,
    experiment_data: ExperimentData,
    inducing_points: Data,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    likelihood: gpytorch.likelihoods.Likelihood,
    seed: int,
    number_of_epochs: int,
    batch_size: int,
    learning_rate_upper: float,
    learning_rate_lower: float,
    number_of_learning_rate_searches: int,
    is_fixed: bool,
    models_path: str,
    early_stopper_patience: float,
    observation_noise: float | None = None,
    plot_title: str | None = None,
    plot_loss_path: str | None = None,
    load_model: bool = True,
) -> Tuple[SVGP, List[float], float]:
    create_directory(models_path)
    best_loss = float("inf")
    losses_history = {}
    model_out = None
    losses_out = None
    best_learning_rate = None
    for i, learning_rate in enumerate(
        np.logspace(
            math.log10(learning_rate_lower),
            math.log10(learning_rate_upper),
            number_of_learning_rate_searches,
        )
    ):
        model_iteration_path = os.path.join(
            models_path,
            f"svgp_{i+1}_of_{number_of_learning_rate_searches}.pth",
        )
        set_seed(seed)
        if os.path.exists(model_iteration_path) and load_model:
            model, losses, _ = load_svgp(
                model_path=model_iteration_path,
                x_induce=inducing_points.x,
                mean=deepcopy(mean),
                kernel=deepcopy(kernel),
                likelihood=deepcopy(likelihood),
                learn_inducing_locations=False if is_fixed else True,
            )
        else:
            model, losses = train_svgp(
                train_data=experiment_data.train,
                inducing_points=inducing_points,
                mean=deepcopy(mean),
                kernel=deepcopy(kernel),
                likelihood=deepcopy(likelihood),
                seed=seed,
                number_of_epochs=number_of_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learn_inducing_locations=False if is_fixed else True,
                learn_kernel_parameters=False if is_fixed else True,
                early_stopper_patience=early_stopper_patience,
                likelihood_noise=observation_noise,
            )
            if model is None:
                continue
            torch.save(
                {
                    "model": model.state_dict(),
                    "losses": losses,
                    "best_learning_rate": best_learning_rate,
                },
                model_iteration_path,
            )
        losses_history[learning_rate] = losses
        loss = losses[-1]
        if loss < best_loss:
            best_loss = loss
            best_learning_rate = learning_rate
            model_out = model
            losses_out = losses
    if plot_loss_path is not None:
        create_directory(plot_loss_path)
        plot_losses(
            losses_history=losses_history,
            title=f"{plot_title} loss ({model_name})"
            if plot_title is not None
            else None,
            save_path=os.path.join(
                plot_loss_path,
                f"{model_name}-losses.png",
            ),
        )
    return model_out, losses_out, best_learning_rate
