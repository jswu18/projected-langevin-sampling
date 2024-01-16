from typing import Dict, List, Tuple, Union

import gpytorch
import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.data import Data, ExperimentData
from src.gps import ExactGP, svGP
from src.gradient_flows.base.base import GradientFlowBase
from src.utils import set_seed

_DATA_COLORS = {
    "train": "tab:blue",
    "test": "tab:orange",
}

_DATA_TRANSPARENCY = {
    "train": 0.3,
    "test": 0.3,
}


def plot_1d_gp_prediction(
    fig: plt.Figure,
    ax: plt.Axes,
    x: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    save_path: str = None,
    title: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    stdev = torch.sqrt(variance)
    ax.fill_between(
        x.reshape(-1),
        (mean - 1.96 * stdev).reshape(-1),
        (mean + 1.96 * stdev).reshape(-1),
        facecolor=(0.85, 0.85, 0.85),
        label="error bound (95%)",
        zorder=0,
    )
    ax.plot(x.reshape(-1), mean.reshape(-1), label="mean", zorder=0, color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return fig, ax
    else:
        return fig, ax


def plot_1d_data(
    fig: plt.Figure,
    ax: plt.Axes,
    data: Data,
    save_path: str = None,
    title: str = None,
    color: str = "tab:blue",
    alpha: float = 0.3,
) -> Tuple[plt.Figure, plt.Axes]:
    if data.name in _DATA_COLORS:
        color = _DATA_COLORS[data.name]
    if data.y is not None:
        ax.scatter(
            data.x,
            data.y,
            label=data.name,
            alpha=alpha,
            color=color,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
        return fig, ax
    else:
        return fig, ax


def plot_1d_experiment_data(
    fig: plt.Figure,
    ax: plt.Axes,
    experiment_data: ExperimentData,
    save_path: str = None,
    title: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    for data in [
        experiment_data.train,
        experiment_data.validation,
        experiment_data.test,
    ]:
        if data is not None:
            fig, ax = plot_1d_data(
                fig=fig,
                ax=ax,
                data=data,
                save_path=None,
                title=None,
                alpha=0.3,
            )
    if experiment_data.full.y_untransformed is not None:
        ax.plot(
            experiment_data.full.x,
            experiment_data.full.y_untransformed.reshape(experiment_data.full.x.shape),
            label="latent",
            color="tab:blue",
            linestyle=(0, (3, 1, 1, 1, 1, 1)),
            linewidth=3,
            alpha=0.5,
        )
        ax.set_ylim([0, 1])
    ax.set_xlim([-2, 2])
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
        return fig, ax
    else:
        return fig, ax


def plot_1d_particle(
    fig: plt.Figure,
    ax: plt.Axes,
    x: torch.Tensor,
    y: torch.Tensor,
    add_label: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    ax.plot(
        x.reshape(-1),
        y.reshape(-1),
        color=[0.1, 0.1, 0.1],
        alpha=0.15,
        zorder=0,
        label="particle" if add_label else None,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    return fig, ax


def plot_1d_pwgf_prediction(
    experiment_data: ExperimentData,
    induce_data: Data,
    x: torch.Tensor,
    predicted_samples: torch.Tensor,
    save_path: str,
    predicted_distribution: gpytorch.distributions.MultivariateNormal = None,
    title: str = None,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    for i in range(induce_data.x.shape[0]):
        plt.axvline(
            x=induce_data.x[i],
            color="tab:blue",
            alpha=0.2,
            label="induce" if i == 0 else None,
            zorder=0,
        )
    ax.autoscale(enable=False)  # turn off autoscale before plotting particles
    if predicted_distribution is not None:
        fig, ax = plot_1d_gp_prediction(
            fig=fig,
            ax=ax,
            x=experiment_data.full.x,
            mean=predicted_distribution.mean.detach(),
            variance=predicted_distribution.variance.detach(),
        )
    for i in range(predicted_samples.shape[1]):
        fig, ax = plot_1d_particle(
            fig=fig,
            ax=ax,
            x=x,
            y=predicted_samples[:, i],
            add_label=i == 0,
        )
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(
        save_path,
    )
    plt.close(fig)


def plot_losses(
    losses_history: Dict[float, List[float]],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, learning_rate in enumerate(losses_history.keys()):
        shade = ((len(losses_history) - i) / len(losses_history)) * 0.8
        ax.plot(
            losses_history[learning_rate],
            label=learning_rate,
            color=[shade, shade, shade],
        )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if len(losses_history) <= 20:
        ax.legend()
    fig.tight_layout()
    fig.savefig(
        save_path,
    )
    plt.close(fig)


def plot_1d_gp_prediction_and_induce_data(
    model: Union[ExactGP, svGP],
    experiment_data: ExperimentData,
    title: str,
    save_path: str,
    induce_data: Data = None,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    for i in range(induce_data.x.shape[0]):
        plt.axvline(
            x=induce_data.x[i],
            color="tab:blue",
            alpha=0.2,
            label="induce" if i == 0 else None,
            zorder=0,
        )
    ax.autoscale(enable=False)  # turn off autoscale before plotting gp prediction
    prediction = model.likelihood(model(experiment_data.full.x))
    if isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood):
        fig, ax = plot_1d_gp_prediction(
            fig=fig,
            ax=ax,
            x=experiment_data.full.x,
            mean=prediction.mean.detach(),
            variance=prediction.variance.detach(),
        )
    elif isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood):
        ax.plot(
            experiment_data.full.x.reshape(-1),
            prediction.mean.detach().reshape(-1),
            label="prediction",
            zorder=0,
            color="black",
        )
        ax.legend()
    else:
        raise NotImplementedError
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_energy_potentials(
    energy_potentials_history: Dict[float, List[float]],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, learning_rate in enumerate(
        sorted(energy_potentials_history.keys(), reverse=True)
    ):
        shade = 0.1 + (i / (max(len(energy_potentials_history) - 1, 1))) * 0.8
        ax.plot(
            learning_rate * np.arange(len(energy_potentials_history[learning_rate])),
            np.log(energy_potentials_history[learning_rate]),
            label=learning_rate,
            color=[shade, shade, shade],
        )
    ax.set_title(title)
    ax.set_xlabel("simulation time")
    ax.set_ylabel("log energy potential")
    plt.ylim(
        [
            0.8 * min([min(np.log(x)) for x in energy_potentials_history.values()]),
            1.5 * max(np.log([x[0] for x in energy_potentials_history.values()])),
        ]
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        save_path,
    )
    plt.close(fig)


def plot_true_versus_predicted(
    y_true: torch.Tensor,
    y_pred: gpytorch.distributions.MultivariateNormal,
    title: str,
    save_path: str,
    error_bar: bool = True,
):
    fig, ax = plt.subplots(figsize=(13, 13))
    if error_bar:
        _, _, bars = plt.errorbar(
            y_true.detach().numpy(),
            y_pred.mean.detach().numpy(),
            yerr=1.96 * y_pred.stddev.detach().numpy(),
            fmt="o",
            elinewidth=0.5,
        )
        [bar.set_alpha(0.5) for bar in bars]
    else:
        plt.scatter(
            y_true.detach().numpy(),
            y_pred.mean.detach().numpy(),
            marker="o",
        )
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    if error_bar:
        ax.set_title(f"{title} (95% confidence interval)")
    else:
        ax.set_title(title)
    axis_lims = [
        min(y_true.min().item(), y_pred.mean.min().item()),
        max(y_true.max().item(), y_pred.mean.max().item()),
    ]
    plt.plot(axis_lims, axis_lims, color="gray", label="target", linestyle="dashed")
    plt.xlim(axis_lims)
    plt.ylim(axis_lims)
    plt.legend()
    fig.tight_layout()
    fig.savefig(
        save_path,
    )
    plt.close(fig)


def animate_1d_pwgf_predictions(
    pwgf: GradientFlowBase,
    number_of_particles: int,
    initial_particles_noise_only: bool,
    seed: int,
    learning_rate: float,
    number_of_epochs: int,
    experiment_data: ExperimentData,
    induce_data: Data,
    x: torch.Tensor,
    title: str,
    save_path: str,
    christmas_colours: bool = False,
    animation_duration: int = 15,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    for i in range(induce_data.x.shape[0]):
        plt.axvline(
            x=induce_data.x[i],
            color="tab:blue",
            alpha=0.2,
            label="induce" if i == 0 else None,
            zorder=0,
        )
    plt.xlim(x.min(), x.max())
    ax.autoscale(enable=False)  # turn off autoscale before plotting particles

    particles = pwgf.initialise_particles(
        number_of_particles=number_of_particles,
        seed=seed,
        noise_only=initial_particles_noise_only,
    )
    predictive_noise = pwgf.sample_predictive_noise(
        x=experiment_data.full.x,
        particles=particles,
    ).detach()
    observation_noise = pwgf.sample_observation_noise(
        number_of_particles=number_of_particles,
        seed=seed,
    ).detach()
    predicted_samples = pwgf.predict_samples(
        x=experiment_data.full.x,
        particles=particles,
        predictive_noise=predictive_noise,
        observation_noise=observation_noise,
    ).detach()
    samples_plotted = [
        ax.plot(
            x.reshape(-1),
            predicted_samples[:, i].reshape(-1),
            color=[0.3, 0.3, 0.3]
            if not christmas_colours
            else ["green", "red", "blue"][i % 3],
            alpha=0.1,
            zorder=0,
            label="particle" if i == 0 else None,
        )[0]
        for i in range(predicted_samples.shape[-1])
    ]
    plt.legend(loc="lower left")

    class ParticleWrapper:
        """
        Wrapper class to allow particles to be updated in the animation function.
        """

        def __init__(self, particles: torch.Tensor):
            self.particles = particles
            self.num_updates = 0

        def update(self, particle_update):
            self.particles += particle_update
            self.num_updates += 1

    particle_wrapper = ParticleWrapper(particles=particles)
    fps = min(number_of_epochs // animation_duration, 30)
    number_of_frames = 1 + number_of_epochs // (
        (number_of_epochs // (animation_duration * fps)) + 1
    )
    progress_bar = tqdm(total=number_of_frames + 1, desc="WGF Particles GIF")

    def animate(iteration: int):
        for _ in range((number_of_epochs // (animation_duration * fps)) + 1):
            particle_wrapper.update(
                pwgf.calculate_particle_update(
                    particles=particle_wrapper.particles,
                    learning_rate=learning_rate,
                )
            )
        _predicted_samples = pwgf.predict_samples(
            x=experiment_data.full.x,
            particles=particle_wrapper.particles,
            predictive_noise=predictive_noise,
            observation_noise=observation_noise,
        ).detach()
        for i in range(_predicted_samples.shape[-1]):
            samples_plotted[i].set_data((x, _predicted_samples[:, i].reshape(-1)))
        ax.set_title(
            f"{title} (simulation time={learning_rate*particle_wrapper.num_updates:.2e}, iteration={particle_wrapper.num_updates})"
        )
        progress_bar.update(n=1)
        return (samples_plotted[0],)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=number_of_frames, interval=50
    )

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(
        fps=fps,
        bitrate=1800,
    )
    ani.save(save_path, writer=writer)
    plt.close()


def animate_1d_gp_predictions(
    experiment_data: ExperimentData,
    induce_data: Data,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    likelihood: Union[
        gpytorch.likelihoods.MultitaskGaussianLikelihood,
        gpytorch.likelihoods.BernoulliLikelihood,
    ],
    seed: int,
    number_of_epochs: int,
    batch_size: int,
    learning_rate: float,
    title: str,
    save_path: str,
    learn_inducing_locations: bool,
    learn_kernel_parameters: bool,
    christmas_colours: bool = False,
    animation_duration: int = 15,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    for i in range(induce_data.x.shape[0]):
        plt.axvline(
            x=induce_data.x[i],
            color="tab:blue",
            alpha=0.2,
            label="induce" if i == 0 else None,
            zorder=0,
        )
    plt.xlim(experiment_data.full.x.min(), experiment_data.full.x.max())
    ax.autoscale(enable=False)  # turn off autoscale before plotting particles

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
        model.likelihood, model, num_data=experiment_data.train.x.shape[0]
    )

    train_dataset = TensorDataset(experiment_data.train.x, experiment_data.train.y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    prediction = model.likelihood(model(experiment_data.full.x))
    mean_prediction = prediction.mean.detach()
    if isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood):
        stdev_prediction = torch.sqrt(prediction.variance.detach())
        fill = ax.fill_between(
            experiment_data.full.x.reshape(-1),
            (mean_prediction - 1.96 * stdev_prediction).reshape(-1),
            (mean_prediction + 1.96 * stdev_prediction).reshape(-1),
            facecolor=(0.8, 0.8, 0.8),
            label="error bound (95%)",
            zorder=0,
        )
        mean_line = ax.plot(
            experiment_data.full.x.reshape(-1),
            mean_prediction.reshape(-1),
            label="mean",
            zorder=0,
        )[0]
    elif isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood):
        mean_line = ax.plot(
            experiment_data.full.x.reshape(-1),
            mean_prediction.reshape(-1),
            label="mean",
            zorder=0,
        )[0]
    else:
        raise NotImplementedError
    plt.legend(loc="lower left")

    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self):
            self.count += 1

    counter = Counter()
    fps = min(number_of_epochs // animation_duration, 30)
    number_of_frames = 1 + number_of_epochs // (
        (number_of_epochs // (animation_duration * fps)) + 1
    )
    progress_bar = tqdm(total=number_of_frames + 1, desc="GP Learning GIF")

    def animate(iteration: int):
        for _ in range((number_of_epochs // (animation_duration * fps)) + 1):
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
            counter.increment()
        _prediction = model.likelihood(model(experiment_data.full.x))
        _mean_prediction = _prediction.mean.detach()
        if isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood):
            _stdev_prediction = torch.sqrt(_prediction.variance.detach())
            path = fill.get_paths()[0]
            verts = path.vertices
            verts[1 : experiment_data.full.x.shape[0] + 1, 1] = (
                _mean_prediction - 1.96 * _stdev_prediction
            ).reshape(-1)
            verts[experiment_data.full.x.shape[0] + 2 : -1, 1] = list(
                (_mean_prediction + 1.96 * _stdev_prediction).reshape(-1)
            )[::-1]
            mean_line.set_data(
                (experiment_data.full.x.reshape(-1), _mean_prediction.reshape(-1))
            )
        elif isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood):
            mean_line.set_data(
                (experiment_data.full.x.reshape(-1), _mean_prediction.reshape(-1))
            )
        else:
            raise NotImplementedError
        ax.set_title(
            f"{title} (simulation time={learning_rate*counter.count:.2e}, iteration={counter.count})"
        )
        if christmas_colours:
            fill.set_color(
                (
                    0.5 * np.sin(iteration / 15) + 0.5,
                    0.5 * np.sin(iteration / 15 + 2 * np.pi / 3) + 0.5,
                    0.5 * np.sin(iteration / 15 + 4 * np.pi / 3) + 0.5,
                )
            )
        progress_bar.update(n=1)
        return (mean_line,)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=number_of_frames, interval=50
    )

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(
        fps=fps,
        bitrate=1800,
    )
    ani.save(save_path, writer=writer)
    plt.close()
