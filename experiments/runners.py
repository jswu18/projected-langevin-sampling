import math
import os
from copy import deepcopy
from typing import List, Optional

import gpytorch
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.data import Data, ExperimentData
from experiments.metrics import calculate_mae, calculate_mse, calculate_nll
from experiments.plotters import (
    plot_1d_gp_prediction_and_induce_data,
    plot_1d_pwgf_prediction,
    plot_losses,
    plot_update_magnitude,
)
from experiments.utils import create_directory
from src.bisection_search import LogBisectionSearch
from src.gps import ExactGP, svGP
from src.gradient_flows.projected_wasserstein import ProjectedWassersteinGradientFlow
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
) -> (ExactGP, List[float]):
    set_seed(seed)
    model = ExactGP(
        mean=mean,
        kernel=kernel,
        x=data.x,
        y=data.y,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
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
        loss = -mll(output, data.y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    return model, losses


def _train_svgp(
    train_data: Data,
    induce_data: Data,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    seed: int,
    number_of_epochs: int,
    batch_size: int,
    learning_rate: float,
    learn_inducing_locations: bool,
    learn_kernel_parameters: bool,
    learn_observation_noise: bool,
    observation_noise: float = None,
) -> (ExactGP, List[float]):
    set_seed(seed)
    model = svGP(
        mean=mean,
        kernel=kernel,
        x_induce=induce_data.x,
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        learn_inducing_locations=learn_inducing_locations,
    )
    model.double()
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    all_params = set(model.parameters())

    if not learn_kernel_parameters:
        all_params -= {model.kernel.base_kernel.raw_lengthscale}
        all_params -= {model.kernel.raw_outputscale}
    if not learn_observation_noise:
        all_params -= {model.likelihood.raw_noise}
        model.likelihood.noise = observation_noise
    model.likelihood.double()
    model.likelihood.train()
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
        desc=f"svGP Epoch ({learn_kernel_parameters=}, {learn_inducing_locations=}, {learn_observation_noise=})",
    )
    losses = []
    for _ in epochs_iter:
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
        output = model(train_data.x)
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
    return Data(
        x=x_induce.double(),
        y=y_induce.double(),
        name="induce",
    )


def optimise_kernel_and_induce_data(
    experiment_data: ExperimentData,
    kernel: gpytorch.kernels.Kernel,
    induce_data_selector: InduceDataSelector,
    seed: int,
    number_of_epochs: int,
    learning_rate: float,
    number_of_iterations: int,
    number_induce_points: int,
    gp_scheme: str = "exact",
    batch_size: int = None,
    plot_1d_iteration_path: str = None,
    plot_loss_path: str = None,
) -> (gpytorch.models.GP, Data):
    losses_history = []
    model_history = []
    train_nll = []
    induce_data_history = []
    for i in range(number_of_iterations):
        induce_data = select_induce_data(
            seed=seed,
            induce_data_selector=induce_data_selector,
            data=experiment_data.train,
            number_induce_points=number_induce_points,
            kernel=kernel,
        )
        induce_data_history.append(induce_data)
        if gp_scheme == "exact":
            model, losses = _train_exact_gp(
                data=induce_data,
                mean=gpytorch.means.ConstantMean(),
                kernel=kernel,
                seed=seed,
                number_of_epochs=number_of_epochs,
                learning_rate=learning_rate,
            )
        elif gp_scheme == "svgp":
            model, losses = _train_svgp(
                train_data=experiment_data.train,
                induce_data=induce_data,
                mean=gpytorch.means.ConstantMean(),
                kernel=kernel,
                seed=seed,
                number_of_epochs=number_of_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                learn_inducing_locations=True,
                learn_kernel_parameters=True,
                learn_observation_noise=True,
            )
        else:
            raise ValueError(f"Unknown GP scheme: {gp_scheme}")
        losses_history.append(losses)
        model_history.append(model)
        nll = calculate_nll(
            prediction=model.forward(experiment_data.train.x),
            y=experiment_data.train.y,
            y_std=experiment_data.y_std,
        )
        train_nll.append(nll)
        kernel = deepcopy(model.kernel)
        if plot_1d_iteration_path is not None:
            create_directory(plot_1d_iteration_path)
            plot_1d_gp_prediction_and_induce_data(
                model=model,
                experiment_data=experiment_data,
                induce_data=induce_data,
                title=f"Inducing Points Selection (iteration {i + 1}, {gp_scheme=}, {nll=:.2f})",
                save_path=os.path.join(
                    plot_1d_iteration_path,
                    f"inducing-points-selection-iteration-{i + 1}.png",
                ),
            )
        if i > 0:
            if any(
                [
                    torch.allclose(induce_data_history[-1].y, induce_data_history[j].y)
                    for j in range(i)
                ]
            ):
                break
    model_out, induce_data_out = (
        model_history[train_nll.index(min(train_nll))],
        induce_data_history[train_nll.index(min(train_nll))],
    )
    if plot_1d_iteration_path is not None:
        create_directory(plot_1d_iteration_path)
        plot_1d_gp_prediction_and_induce_data(
            model=model_out,
            experiment_data=experiment_data,
            induce_data=induce_data_out,
            title=f"Inducing Points Selection ({gp_scheme=})",
            save_path=os.path.join(
                plot_1d_iteration_path,
                f"inducing-points-selection.png",
            ),
        )
    if plot_loss_path is not None:
        create_directory(plot_loss_path)
        plot_losses(
            losses_history=losses_history,
            title=f"Inducing Points Selection ({gp_scheme=})",
            save_path=os.path.join(
                plot_loss_path,
                "inducing-points-selection-losses.png",
            ),
        )
    return model_out, induce_data_out


def learn_subsample_gps(
    experiment_data: ExperimentData,
    kernel: gpytorch.kernels.Kernel,
    subsample_size: int,
    seed: int,
    number_of_epochs: int,
    learning_rate: float,
    number_of_iterations: int,
    plot_1d_subsample_path: str = None,
    plot_loss_path: str = None,
) -> List[gpytorch.models.GP]:
    set_seed(seed)
    if subsample_size > len(experiment_data.train.x):
        model, losses = _train_exact_gp(
            data=experiment_data.train,
            mean=gpytorch.means.ConstantMean(),
            kernel=deepcopy(kernel),
            seed=seed,
            number_of_epochs=number_of_epochs,
            learning_rate=learning_rate,
        )
        losses_history = [losses]
        models = [model]
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
    else:
        models = []
        knn = NearestNeighbors(
            n_neighbors=subsample_size,
            p=2,
        )
        knn.fit(X=experiment_data.train.x, y=experiment_data.train.y)
        losses_history = []
        for i in range(number_of_iterations):
            x_sample = sample_point(x=experiment_data.train.x)
            subsample_indices = knn.kneighbors(
                X=x_sample,
                return_distance=False,
            )
            data_subsample = Data(
                x=experiment_data.train.x[subsample_indices],
                y=experiment_data.train.y[subsample_indices],
            )
            model, losses = _train_exact_gp(
                data=data_subsample,
                mean=gpytorch.means.ConstantMean(),
                kernel=deepcopy(kernel),
                seed=seed,
                number_of_epochs=number_of_epochs,
                learning_rate=learning_rate,
            )
            models.append(model)
            losses_history.append(losses)
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
    particle_name: str,
    kernel: gpytorch.kernels.Kernel,
    experiment_data: ExperimentData,
    induce_data: Data,
    number_of_particles: int,
    number_of_epochs: int,
    learning_rate_upper: float,
    learning_rate_lower: float,
    number_of_learning_rate_searches: int,
    max_particle_magnitude: float,
    observation_noise: float,
    jitter: float,
    seed: int,
    plot_title: str = None,
    plot_particles_path: str = None,
    plot_update_magnitude_path: str = None,
) -> ProjectedWassersteinGradientFlow:
    gradient_flow_kernel = GradientFlowKernel(
        base_kernel=kernel,
        approximation_samples=experiment_data.train.x,
    )
    pwgf = ProjectedWassersteinGradientFlow(
        number_of_particles=number_of_particles,
        seed=seed,
        kernel=gradient_flow_kernel,
        x_induce=induce_data.x,
        y_induce=induce_data.y,
        x_train=experiment_data.train.x,
        y_train=experiment_data.train.y,
        jitter=jitter,
        observation_noise=observation_noise,
    )
    if plot_particles_path is not None:
        create_directory(plot_particles_path)
        plot_1d_pwgf_prediction(
            experiment_data=experiment_data,
            induce_data=induce_data,
            x=experiment_data.full.x,
            predicted_samples=pwgf.predict_samples(
                x=experiment_data.full.x,
                include_observation_noise=False,
            ).detach(),
            title=f"{plot_title} (initial particles)"
            if plot_title is not None
            else None,
            save_path=os.path.join(
                plot_particles_path,
                f"particles-initial-{particle_name}.png",
            ),
        )
    particles_out = None
    best_mae = float("inf")
    best_mse = float("inf")
    best_nll = float("inf")
    searches_iter = tqdm(range(number_of_learning_rate_searches), desc="WGF LR Search")
    update_log_magnitude_history = {}
    learning_rate_bisection_search = LogBisectionSearch(
        lower=learning_rate_lower,
        upper=learning_rate_upper,
        soft_update=True,
    )
    for _ in searches_iter:
        pwgf.reset_particles(seed=seed)
        update_log_magnitudes = []
        set_seed(seed)
        for i in range(number_of_epochs):
            particle_update = pwgf.update(
                learning_rate=torch.tensor(learning_rate_bisection_search.current),
            )
            update_log_magnitudes.append(
                float(
                    torch.log(torch.norm(particle_update, dim=1).mean()).detach().item()
                )
            )
            if (
                torch.isnan(pwgf.particles).any()
                or torch.max(torch.abs(pwgf.particles)) > max_particle_magnitude
            ).item():
                searches_iter.set_postfix(
                    best_mae=best_mae,
                    best_mse=best_mse,
                    best_nll=best_nll,
                    lr=learning_rate_bisection_search.current,
                    upper=learning_rate_bisection_search.upper,
                    lower=learning_rate_bisection_search.lower,
                )
                learning_rate_bisection_search.update_upper()
                break
        else:  # only executed if the inner loop did NOT break
            prediction = pwgf.predict(
                x=experiment_data.train.x,
            )
            nll = calculate_nll(
                prediction=prediction,
                y=experiment_data.train.y,
                y_std=experiment_data.y_std,
            )
            mse = calculate_mse(
                prediction=prediction,
                y=experiment_data.train.y,
            )
            mae = calculate_mae(
                prediction=prediction,
                y=experiment_data.train.y,
            )
            if nll < best_nll:
                best_nll = nll
                best_mae = mae
                best_mse = mse
                particles_out = deepcopy(pwgf.particles.detach())
            searches_iter.set_postfix(
                best_mae=best_mae,
                best_mse=best_mse,
                best_nll=best_nll,
                lr=learning_rate_bisection_search.current,
                upper=learning_rate_bisection_search.upper,
                lower=learning_rate_bisection_search.lower,
            )
            update_log_magnitude_history[
                learning_rate_bisection_search.current
            ] = update_log_magnitudes
            # soft bound update
            learning_rate_bisection_search.update_lower()
    pwgf.particles = particles_out
    if plot_particles_path is not None:
        create_directory(plot_particles_path)
        plot_1d_pwgf_prediction(
            experiment_data=experiment_data,
            induce_data=induce_data,
            x=experiment_data.full.x,
            predicted_samples=pwgf.predict_samples(
                x=experiment_data.full.x,
                include_observation_noise=False,
            ).detach(),
            title=f"{plot_title} (learned particles, {best_mae=:.2f}, {best_mse=:.2f}, {best_nll=:.2f})"
            if plot_title is not None
            else None,
            save_path=os.path.join(
                plot_particles_path,
                f"particles-learned-{particle_name}.png",
            ),
        )
    if plot_update_magnitude_path is not None:
        create_directory(plot_update_magnitude_path)
        plot_update_magnitude(
            update_log_magnitude_history=update_log_magnitude_history,
            title=f"{plot_title} (update magnitude)"
            if plot_title is not None
            else None,
            save_path=os.path.join(
                plot_update_magnitude_path,
                f"update-magnitude-{particle_name}.png",
            ),
        )
    return pwgf


def pwgf_observation_noise_search(
    data: Data,
    model: ProjectedWassersteinGradientFlow,
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
            ),
            y=data.y,
            y_std=y_std,
        )
        model.observation_noise = torch.tensor(bisection_search.upper)
        set_seed(0)
        upper_nll = calculate_nll(
            prediction=model.predict(
                x=data.x,
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
    experiment_data: ExperimentData,
    induce_data: Data,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    seed: int,
    number_of_epochs: int,
    batch_size: int,
    learning_rate_upper: float,
    learning_rate_lower: float,
    number_of_learning_rate_searches: int,
    is_fixed: bool,
    observation_noise: float = None,
    plot_title: Optional[str] = None,
    plot_1d_path: Optional[str] = None,
    plot_loss_path: Optional[str] = None,
) -> (svGP, List[float]):
    model_name = "fixed-svgp" if is_fixed else "svgp"
    best_nll = float("inf")
    best_mse = float("inf")
    best_mae = float("inf")
    best_loss = float("inf")
    losses_history = []
    model_out = None
    for log_learning_rate in torch.linspace(
        math.log(learning_rate_lower),
        math.log(learning_rate_upper),
        number_of_learning_rate_searches,
    ):
        learning_rate = math.exp(log_learning_rate)
        set_seed(seed)
        model, losses = _train_svgp(
            train_data=experiment_data.train,
            induce_data=induce_data,
            mean=deepcopy(mean),
            kernel=deepcopy(kernel),
            seed=seed,
            number_of_epochs=number_of_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learn_inducing_locations=False if is_fixed else True,
            learn_kernel_parameters=False if is_fixed else True,
            learn_observation_noise=False if is_fixed else True,
            observation_noise=observation_noise,
        )
        prediction = model.forward(experiment_data.train.x)
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
        losses_history.append(losses)
        if nll < best_nll:
            best_nll = nll
            best_mae = mae
            best_mse = mse
            best_loss = float(losses[-1])
            model_out = model
    if plot_1d_path is not None:
        create_directory(plot_1d_path)
        plot_1d_gp_prediction_and_induce_data(
            model=model_out,
            experiment_data=experiment_data,
            induce_data=induce_data
            if is_fixed
            else None,  # induce data can't be visualised if it's learned by the model
            title=f"{plot_title}, ({model_name}, {best_mae=:.2f}, {best_mse=:.2f}, {best_nll=:.2f})"
            if plot_title is not None
            else None,
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
    return model_out


def construct_average_ard_kernel(
    kernels: List[gpytorch.kernels.Kernel],
) -> gpytorch.kernels.Kernel:
    kernel = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(
            ard_num_dims=kernels[0].base_kernel.ard_num_dims,
        )
    )
    kernel.base_kernel.lengthscale = torch.concat(
        tensors=[k.base_kernel.lengthscale for k in kernels],
    ).mean(dim=0)
    kernel.outputscale = torch.tensor(
        data=[k.outputscale for k in kernels],
    ).mean(dim=0)
    return kernel


def construct_average_gaussian_likelihood(
    likelihoods: List[gpytorch.likelihoods.GaussianLikelihood],
) -> gpytorch.likelihoods.GaussianLikelihood:
    average_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    average_likelihood.noise = torch.tensor(
        [likelihood.noise for likelihood in likelihoods]
    ).mean()
    return average_likelihood
