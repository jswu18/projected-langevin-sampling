import math
import os
from copy import deepcopy
from typing import List, Optional, Union

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
    animate_1d_pwgf_predictions,
    plot_1d_gp_prediction_and_induce_data,
    plot_1d_pwgf_prediction,
    plot_energy_potentials,
    plot_losses,
)
from experiments.utils import create_directory
from src.bisection_search import LogBisectionSearch
from src.gps import ExactGP, svGP
from src.gradient_flows.base.base import GradientFlowBase
from src.induce_data_selectors import InduceDataSelector
from src.kernels import GradientFlowKernel
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
    model.double()
    likelihood.double()
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


def _train_svgp(
    train_data: Data,
    induce_data: Data,
    mean: gpytorch.means.Mean,
    kernel: GradientFlowKernel,
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
    likelihood_noise: Optional[float] = None,
) -> (ExactGP, List[float]):
    set_seed(seed)
    model = svGP(
        mean=mean,
        kernel=kernel,
        x_induce=induce_data.x,
        likelihood=likelihood,
        learn_inducing_locations=learn_inducing_locations,
    )
    model.double()
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    all_params = set(model.parameters())

    if not learn_kernel_parameters:
        all_params -= {model.kernel.base_kernel.base_kernel.raw_lengthscale}
        all_params -= {model.kernel.base_kernel.raw_outputscale}
    model.likelihood.double()
    model.likelihood.train()
    if likelihood_noise is not None:
        model.likelihood.noise_covar.noise.data.fill_(likelihood_noise)
        model.likelihood.noise_covar.raw_noise.requires_grad_(False)
    if torch.cuda.is_available():
        model.likelihood = model.likelihood.cuda()
    model.likelihood.train()

    optimizer = torch.optim.Adam(
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
        desc=f"svGP Epoch ({learn_kernel_parameters=}, {learn_inducing_locations=})",
    )
    losses = []
    for _ in epochs_iter:
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
        output = model.forward(train_data.x)
        losses.append(-mll(output, train_data.y).item())
    model.eval()
    return model, losses


def select_induce_data(
    seed: int,
    induce_data_selector: InduceDataSelector,
    data: Data,
    number_induce_points: int,
    kernel: gpytorch.kernels.Kernel,
) -> Data:
    set_seed(seed)
    x_induce, induce_indices = induce_data_selector(
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
        x=x_induce.double(),
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
    )
    return Data(
        x=data.x[subsample_indices].double(),
        y=data.y[subsample_indices].double(),
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
    if subsample_size > len(experiment_data.train.x):
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
                plot_1d_gp_prediction_and_induce_data(
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


def train_projected_wasserstein_gradient_flow(
    pwgf: GradientFlowBase,
    number_of_particles: int,
    particle_name: str,
    experiment_data: ExperimentData,
    induce_data: Data,
    simulation_duration: float,
    seed: int,
    observation_noise_upper: float = 0,
    observation_noise_lower: float = 0,
    number_of_observation_noise_searches: int = 0,
    plot_title: str = None,
    plot_particles_path: str = None,
    animate_1d_path: str = None,
    plot_update_magnitude_path: str = None,
    christmas_colours: bool = False,
    metric_to_minimise: str = "nll",
    initial_particles_noise_only: bool = False,
) -> torch.Tensor:
    particles_out = pwgf.initialise_particles(
        number_of_particles=number_of_particles,
        seed=seed,
        noise_only=initial_particles_noise_only,
    )
    if plot_particles_path is not None:
        create_directory(plot_particles_path)
        predicted_distribution = pwgf.predict(
            x=experiment_data.full.x,
            particles=particles_out,
        )
        plot_1d_pwgf_prediction(
            experiment_data=experiment_data,
            induce_data=induce_data,
            x=experiment_data.full.x,
            predicted_samples=pwgf.predict_samples(
                x=experiment_data.full.x,
                particles=particles_out,
            ).detach(),
            predicted_distribution=predicted_distribution
            if isinstance(
                predicted_distribution, gpytorch.distributions.MultivariateNormal
            )
            else None,
            title=f"{plot_title} (initial particles)"
            if plot_title is not None
            else None,
            save_path=os.path.join(
                plot_particles_path,
                f"particles-initial-{particle_name}.png",
            ),
        )
    best_energy_potential, best_mae, best_mse, best_nll = (
        float("inf"),
        float("inf"),
        float("inf"),
        float("inf"),
    )
    best_lr = None
    energy_potentials_history = {}
    for learning_rate in tqdm(
        np.logspace(-3, np.log10(simulation_duration / 1e6), 5),
        desc=f"WGF LR Search {particle_name}",
    ):
        number_of_epochs = int(simulation_duration / learning_rate)
        particles = pwgf.initialise_particles(
            number_of_particles=number_of_particles,
            seed=seed,
            noise_only=initial_particles_noise_only,
        )
        energy_potentials = []
        early_stopper = EarlyStopper()
        set_seed(seed)
        for i in range(number_of_epochs):
            particle_update = pwgf.calculate_particle_update(
                particles=particles,
                learning_rate=learning_rate,
            )
            particles += particle_update
            energy_potential = pwgf.calculate_energy_potential(particles=particles)
            if early_stopper.should_stop(
                loss=energy_potential, learning_rate=learning_rate
            ):
                break
            energy_potentials.append(energy_potential)
        if energy_potentials and np.isfinite(particles).all():
            energy_potentials_history[learning_rate] = energy_potentials
            prediction = pwgf.predict(
                x=experiment_data.train.x,
                particles=particles,
            )
            nll = calculate_nll(
                prediction=prediction,
                y=experiment_data.train.y.double(),
                y_std=experiment_data.y_std,
            )
            mse = calculate_mse(
                prediction=prediction,
                y=experiment_data.train.y.double(),
            )
            mae = calculate_mae(
                prediction=prediction,
                y=experiment_data.train.y.double(),
            )
            # if metric_to_minimise not in ["nll", "mse", "mae"]:
            #     raise ValueError(f"Unknown metric to minimise: {metric_to_minimise}")
            # elif (
            #     (metric_to_minimise == "nll" and nll < best_nll)
            #     or (metric_to_minimise == "mse" and mse < best_mse)
            #     or (metric_to_minimise == "mae" and mae < best_mae)
            # ):
            if energy_potentials[-1] < best_energy_potential:
                best_nll, best_mae, best_mse, best_lr = (
                    nll,
                    mae,
                    mse,
                    learning_rate,
                )
                best_energy_potential = energy_potentials[-1]
                particles_out = deepcopy(particles.detach())
                best_lr = learning_rate
    particles = particles_out
    if number_of_observation_noise_searches:
        pwgf.observation_noise = pwgf_observation_noise_search(
            data=experiment_data.train,
            model=pwgf,
            particles=particles_out,
            observation_noise_upper=observation_noise_upper,
            observation_noise_lower=observation_noise_lower,
            number_of_searches=number_of_observation_noise_searches,
            y_std=experiment_data.y_std,
        )
    if plot_particles_path is not None:
        create_directory(plot_particles_path)
        predicted_distribution = pwgf.predict(
            x=experiment_data.full.x,
            particles=particles_out,
        )
        plot_1d_pwgf_prediction(
            experiment_data=experiment_data,
            induce_data=induce_data,
            x=experiment_data.full.x,
            predicted_samples=pwgf.predict_samples(
                x=experiment_data.full.x,
                particles=particles,
            ).detach(),
            predicted_distribution=predicted_distribution
            if isinstance(
                predicted_distribution, gpytorch.distributions.MultivariateNormal
            )
            else None,
            title=f"{plot_title} (learned particles)"
            if plot_title is not None
            else None,
            save_path=os.path.join(
                plot_particles_path,
                f"particles-learned-{particle_name}.png",
            ),
        )
    if energy_potentials_history and plot_update_magnitude_path is not None:
        create_directory(plot_update_magnitude_path)
        plot_energy_potentials(
            energy_potentials_history=energy_potentials_history,
            title=f"{plot_title} (energy potentials)"
            if plot_title is not None
            else None,
            save_path=os.path.join(
                plot_update_magnitude_path,
                f"energy-potential-{particle_name}.png",
            ),
        )
    if best_lr and animate_1d_path is not None:
        animate_1d_pwgf_predictions(
            pwgf=pwgf,
            seed=seed,
            number_of_particles=number_of_particles,
            initial_particles_noise_only=initial_particles_noise_only,
            learning_rate=best_lr,
            number_of_epochs=int(simulation_duration / best_lr),
            experiment_data=experiment_data,
            induce_data=induce_data,
            x=experiment_data.full.x,
            title=plot_title,
            save_path=os.path.join(
                plot_particles_path,
                f"{particle_name}.gif",
            ),
            christmas_colours=christmas_colours,
        )
    return particles


def pwgf_observation_noise_search(
    data: Data,
    model: GradientFlowBase,
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
    searches_iter = tqdm(range(number_of_searches), desc="PWGF Noise Search")
    for _ in searches_iter:
        model.observation_noise = torch.tensor(bisection_search.lower)
        set_seed(0)
        lower_nll = calculate_nll(
            prediction=model.predict(
                x=data.x,
                particles=particles,
            ),
            y=data.y,
            y_std=y_std,
        )
        model.observation_noise = torch.tensor(bisection_search.upper)
        set_seed(0)
        upper_nll = calculate_nll(
            prediction=model.predict(
                x=data.x,
                particles=particles,
            ),
            y=data.y,
            y_std=y_std,
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


def train_svgp(
    model_name: str,
    experiment_data: ExperimentData,
    induce_data: Data,
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
    observation_noise: Optional[float] = None,
    plot_title: Optional[str] = None,
    plot_1d_path: Optional[str] = None,
    animate_1d_path: Optional[str] = None,
    plot_loss_path: Optional[str] = None,
    christmas_colours: bool = False,
) -> (svGP, List[float]):
    create_directory(models_path)
    best_nll = float("inf")
    best_mse = float("inf")
    best_mae = float("inf")
    best_loss = float("inf")
    losses_history = {}
    model_out = None
    losses_out = None
    best_learning_rate = None
    for i, log_learning_rate in enumerate(
        torch.linspace(
            math.log10(learning_rate_lower),
            math.log10(learning_rate_upper),
            number_of_learning_rate_searches,
        )
    ):
        model_iteration_path = os.path.join(
            models_path,
            f"svgp_{i+1}_of_{number_of_learning_rate_searches}.pth",
        )
        learning_rate = math.exp(log_learning_rate)
        set_seed(seed)
        if os.path.exists(model_iteration_path):
            model, losses = load_svgp(
                model_path=model_iteration_path,
                x_induce=induce_data.x,
                mean=deepcopy(mean),
                kernel=deepcopy(kernel),
                likelihood=deepcopy(likelihood),
                learn_inducing_locations=False if is_fixed else True,
            )
        else:
            model, losses = _train_svgp(
                train_data=experiment_data.train,
                induce_data=induce_data,
                mean=deepcopy(mean),
                kernel=deepcopy(kernel),
                likelihood=deepcopy(likelihood),
                seed=seed,
                number_of_epochs=number_of_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learn_inducing_locations=False if is_fixed else True,
                learn_kernel_parameters=False if is_fixed else True,
                likelihood_noise=observation_noise,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "losses": losses,
                },
                model_iteration_path,
            )
        losses_history[learning_rate] = losses
        set_seed(seed)
        prediction = model.likelihood(model(experiment_data.train.x))
        nll = calculate_nll(
            prediction=prediction,
            y=experiment_data.train.y,
            y_std=experiment_data.y_std,
        )
        mae = calculate_mae(
            prediction=prediction,
            y=experiment_data.train.y,
        )
        mse = calculate_mse(
            prediction=prediction,
            y=experiment_data.train.y,
        )
        loss = losses[-1]
        if loss < best_loss:
            best_nll = nll
            best_mae = mae
            best_mse = mse
            best_loss = loss
            best_learning_rate = learning_rate
            model_out = model
            losses_out = losses
    if plot_1d_path is not None:
        create_directory(plot_1d_path)
        plot_1d_gp_prediction_and_induce_data(
            model=model_out,
            experiment_data=experiment_data,
            induce_data=induce_data
            if is_fixed
            else None,  # induce data can't be visualised if it's learned by the model
            title=f"{plot_title} ({model_name})" if plot_title is not None else None,
            save_path=os.path.join(
                plot_1d_path,
                f"{model_name}.png",
            ),
        )
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
    if animate_1d_path is not None:
        animate_1d_gp_predictions(
            experiment_data=experiment_data,
            induce_data=induce_data,
            mean=deepcopy(mean),
            kernel=deepcopy(kernel),
            likelihood=deepcopy(likelihood),
            seed=seed,
            number_of_epochs=number_of_epochs,
            batch_size=batch_size,
            learning_rate=best_learning_rate,
            title=f"{plot_title} ({model_name})" if plot_title is not None else None,
            save_path=os.path.join(
                plot_1d_path,
                f"{model_name}.gif",
            ),
            learn_inducing_locations=False if is_fixed else True,
            learn_kernel_parameters=False if is_fixed else True,
            christmas_colours=christmas_colours,
        )
    return model_out, losses_out
