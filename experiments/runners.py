import math
import os
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.data import Data, ExperimentData
from experiments.early_stopper import EarlyStopper
from experiments.loaders import load_ard_exact_gp_model, load_svgp
from experiments.metrics import calculate_mae, calculate_mse, calculate_nll
from experiments.plotters import (
    animate_1d_gp_predictions,
    animate_1d_pls_predictions,
    animate_1d_pls_untransformed_predictions,
    plot_1d_gp_prediction_and_inducing_points,
    plot_1d_pls_prediction,
    plot_1d_pls_prediction_histogram,
    plot_energy_potentials,
    plot_losses,
)
from experiments.utils import create_directory
from src.bisection_search import LogBisectionSearch
from src.gps import ExactGP, svGP
from src.inducing_point_selectors import InducingPointSelector
from src.kernels import PLSKernel
from src.projected_langevin_sampling.base.base import PLSBase
from src.samplers import sample_point
from src.utils import set_seed


def _train_exact_gp(
    data: Data,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    seed: int,
    number_of_epochs: int,
    learning_rate: float,
    likelihood: gpytorch.likelihoods.Likelihood,
) -> (ExactGP, List[float]):
    set_seed(seed)
    model = ExactGP(
        mean=mean,
        kernel=kernel,
        x=data.x,
        y=data.y,
        likelihood=likelihood,
    )
    likelihood = model.likelihood
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    epochs_iter = tqdm(range(number_of_epochs), desc="Exact GP Epoch")
    losses = []
    for _ in epochs_iter:
        optimizer.zero_grad()
        output = model(data.x)
        loss = -mll(output, data.y).sum()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    return model, losses


def select_inducing_points(
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


def _load_subsample_data(
    data: Data,
    subsample_size: int,
):
    if subsample_size > len(data.x):
        return data
    knn = NearestNeighbors(
        n_neighbors=subsample_size,
        p=2,
    )
    knn.fit(X=data.x, y=data.y)
    x_sample = sample_point(x=data.x)
    subsample_indices = knn.kneighbors(
        X=x_sample,
        return_distance=False,
    ).flatten()
    return Data(
        x=data.x[subsample_indices],
        y=data.y[..., subsample_indices],
    )


def learn_subsample_gps(
    experiment_data: ExperimentData,
    kernel: gpytorch.kernels.Kernel,
    likelihood: gpytorch.likelihoods.Likelihood,
    subsample_size: int,
    seed: int,
    number_of_epochs: int,
    learning_rate: float,
    number_of_iterations: int,
    model_path: str,
    data_path: str,
    plot_1d_subsample_path: str = None,
    plot_loss_path: str = None,
    number_of_classes: int = 1,
) -> List[gpytorch.models.GP]:
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
            data = _load_subsample_data(
                data=experiment_data.train,
                subsample_size=subsample_size,
            )
            model, losses = _train_exact_gp(
                data=data,
                mean=mean,
                kernel=deepcopy(kernel),
                seed=seed,
                number_of_epochs=number_of_epochs,
                learning_rate=learning_rate,
                likelihood=likelihood,
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


def plot_pls_1d_particles(
    pls: PLSBase,
    particles: torch.Tensor,
    particle_name: str,
    experiment_data: ExperimentData,
    inducing_points: Data,
    plot_particles_path: str,
    plot_title: str = None,
) -> None:
    create_directory(plot_particles_path)
    predicted_distribution = pls.predict(
        x=experiment_data.full.x,
        particles=particles,
    )
    predicted_samples = pls.predict_samples(
        x=experiment_data.full.x,
        particles=particles,
    ).detach()
    plot_1d_pls_prediction(
        experiment_data=experiment_data,
        x=experiment_data.full.x,
        predicted_samples=predicted_samples,
        predicted_distribution=predicted_distribution,
        title=f"{plot_title}" if plot_title is not None else None,
        save_path=os.path.join(
            plot_particles_path,
            f"particles-{particle_name}.png",
        ),
    )
    if experiment_data.full.y_untransformed is not None:
        untransformed_predicted_samples = pls.predict_untransformed_samples(
            x=experiment_data.full.x,
            particles=particles,
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


def animate_pls_1d_particles(
    pls: PLSBase,
    number_of_particles: int,
    particle_name: str,
    experiment_data: ExperimentData,
    seed: int,
    best_lr: float,
    number_of_epochs: int,
    animate_1d_path: str,
    plot_title: str = None,
    animate_1d_untransformed_path: str = None,
    christmas_colours: bool = False,
    initial_particles_noise_only: bool = False,
):
    if best_lr is None:
        return
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
                animate_1d_path,
                f"untransformed-{particle_name}.gif",
            ),
            christmas_colours=christmas_colours,
        )


def train_pls(
    pls: PLSBase,
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
    observation_noise_upper: float = 0,
    observation_noise_lower: float = 0,
    number_of_observation_noise_searches: int = 0,
    plot_title: str = None,
    plot_energy_potential_path: str = None,
    metric_to_minimise: str = "nll",
) -> Tuple[torch.Tensor, float, int]:
    best_energy_potential, best_mae, best_mse, best_nll = (
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
    )
    best_lr = None
    energy_potentials_history = {}
    step_sizes = np.logspace(
        np.log10(step_size_upper),
        np.log10(simulation_duration / maximum_number_of_steps),
        number_of_step_searches,
    )
    particles_out = particles.detach().clone()
    for i, step_size in enumerate(
        tqdm(
            step_sizes,
            desc=f"PLS Step Size Search {particle_name}",
        )
    ):
        number_of_epochs = int(simulation_duration / step_size)
        particles_i = particles.detach().clone()
        energy_potentials = []
        early_stopper = EarlyStopper(patience=early_stopper_patience)
        set_seed(seed)
        for _ in range(number_of_epochs):
            particle_update = pls.calculate_particle_update(
                particles=particles_i,
                step_size=step_size,
            )
            particles_i += particle_update
            energy_potential = pls.calculate_energy_potential(particles=particles_i)
            if early_stopper.should_stop(loss=energy_potential, step_size=step_size):
                break
            energy_potentials.append(energy_potential)
        if energy_potentials and np.isfinite(particles_i).all():
            energy_potentials_history[step_size] = energy_potentials
            prediction = pls.predict(
                x=experiment_data.train.x,
                particles=particles_i,
            )
            nll = calculate_nll(
                prediction=prediction,
                y=experiment_data.train.y,
            )
            mse = calculate_mse(
                prediction=prediction,
                y=experiment_data.train.y,
            )
            mae = calculate_mae(
                prediction=prediction,
                y=experiment_data.train.y,
            )
            if metric_to_minimise not in ["nll", "mse", "mae", "loss"]:
                raise ValueError(f"Unknown metric to minimise: {metric_to_minimise}")
            elif (
                (metric_to_minimise == "nll" and nll < best_nll)
                or (metric_to_minimise == "mse" and mse < best_mse)
                or (metric_to_minimise == "mae" and mae < best_mae)
                or (
                    metric_to_minimise == "loss"
                    and energy_potentials[-1] < best_energy_potential
                )
            ):
                best_nll, best_mae, best_mse, best_lr = (
                    nll,
                    mae,
                    mse,
                    step_size,
                )
                best_energy_potential = energy_potentials[-1]
                particles_out = deepcopy(particles_i.detach())
                best_lr = step_size
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
    if number_of_observation_noise_searches:
        pls.observation_noise = pls_observation_noise_search(
            data=experiment_data.train,
            model=pls,
            particles=particles_out,
            observation_noise_upper=observation_noise_upper,
            observation_noise_lower=observation_noise_lower,
            number_of_searches=number_of_observation_noise_searches,
            y_std=experiment_data.y_std,
        )
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


def pls_observation_noise_search(
    data: Data,
    model: PLSBase,
    particles: torch.Tensor,
    observation_noise_upper: float,
    observation_noise_lower: float,
    number_of_searches: int,
    y_std: float,
) -> float:
    bisection_search = LogBisectionSearch(
        lower=observation_noise_lower,
        upper=observation_noise_upper,
        soft_update=True,
    )
    searches_iter = tqdm(range(number_of_searches), desc="PLS Noise Search")
    for _ in searches_iter:
        model.observation_noise = torch.tensor(bisection_search.lower)
        set_seed(0)
        lower_nll = calculate_nll(
            prediction=model.predict(
                x=data.x,
                particles=particles,
            ),
            y=data.y,
        )
        model.observation_noise = torch.tensor(bisection_search.upper)
        set_seed(0)
        upper_nll = calculate_nll(
            prediction=model.predict(
                x=data.x,
                particles=particles,
            ),
            y=data.y,
        )
        searches_iter.set_postfix(
            lower_nll=lower_nll,
            upper_nll=upper_nll,
            current=bisection_search.current,
            upper=bisection_search.upper,
            lower=bisection_search.lower,
        )
        if lower_nll < upper_nll:
            bisection_search.update_upper()
        else:
            bisection_search.update_lower()
    return bisection_search.current


def _train_svgp(
    train_data: Data,
    inducing_points: Data,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    likelihood: Union[
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.likelihoods.BernoulliLikelihood,
    ],
    seed: int,
    number_of_epochs: int,
    batch_size: int,
    learning_rate: float,
    learn_inducing_locations: bool,
    learn_kernel_parameters: bool,
    early_stopper_patience: float,
    likelihood_noise: Optional[float] = None,
) -> (ExactGP, List[float]):
    set_seed(seed)
    model = svGP(
        mean=mean,
        kernel=kernel,
        x_induce=inducing_points.x,
        likelihood=likelihood,
        learn_inducing_locations=learn_inducing_locations,
    )
    early_stopper = EarlyStopper(patience=early_stopper_patience)
    all_params = set(model.parameters())
    if not learn_kernel_parameters:
        if isinstance(model.kernel, PLSKernel):
            all_params -= {model.kernel.base_kernel.base_kernel.raw_lengthscale}
            all_params -= {model.kernel.base_kernel.raw_outputscale}
        else:
            all_params -= {model.kernel.base_kernel.raw_lengthscale}
            all_params -= {model.kernel.raw_outputscale}
    if likelihood_noise is not None:
        model.likelihood.noise_covar.noise.data.fill_(likelihood_noise)
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

    train_dataset = TensorDataset(train_data.x, train_data.y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    epochs_iter = tqdm(
        range(number_of_epochs),
        desc=f"svGP Epoch ({learn_kernel_parameters=}, {learn_inducing_locations=}, {learning_rate=})",
    )
    losses = []
    for _ in epochs_iter:
        try:
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = -mll(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
            loss = -mll(model(train_data.x), train_data.y).item()
            if early_stopper.should_stop(loss=loss, step_size=learning_rate):
                break
            losses.append(loss)
        except ValueError as e:
            print(e)
            print("Continuing...")
            return None, None
    model.eval()
    return model, losses


def train_svgp(
    model_name: str,
    experiment_data: ExperimentData,
    inducing_points: Data,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    likelihood: Union[
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.likelihoods.BernoulliLikelihood,
    ],
    seed: int,
    number_of_epochs: int,
    batch_size: int,
    learning_rate_upper: float,
    learning_rate_lower: float,
    number_of_learning_rate_searches: int,
    is_fixed: bool,
    models_path: str,
    early_stopper_patience: float,
    observation_noise: Optional[float] = None,
    plot_title: Optional[str] = None,
    plot_loss_path: Optional[str] = None,
    load_model: bool = True,
) -> Tuple[svGP, List[float], float]:
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
            model, losses = _train_svgp(
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
