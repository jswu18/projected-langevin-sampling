from typing import Dict, List, Optional, Tuple, Union

import gpytorch
import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.data import Data, ExperimentData, ProblemType
from src.gps import ExactGP, svGP
from src.kernels import PLSKernel
from src.projected_langevin_sampling.base.base import PLSBase
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
    variance: Optional[torch.Tensor] = None,
    save_path: str = None,
    title: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if variance is not None:
        stdev = torch.sqrt(variance)
        ax.fill_between(
            x.reshape(-1),
            (mean - 1.96 * stdev).reshape(-1),
            (mean + 1.96 * stdev).reshape(-1),
            facecolor=(0.9, 0.9, 0.9),
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
    is_sample_untransformed: bool = False,
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
                data=data
                if not is_sample_untransformed
                else Data(
                    x=data.x,
                    y=data.y_untransformed,
                    name=data.name,
                ),
                save_path=None,
                title=None,
                alpha=0.3,
            )
    if not is_sample_untransformed and experiment_data.full.y_untransformed is not None:
        ax.plot(
            experiment_data.full.x,
            experiment_data.full.y_untransformed.reshape(experiment_data.full.x.shape),
            label="latent",
            color="tab:blue",
            linestyle=(0, (3, 1, 1, 1, 1, 1)),
            linewidth=3,
            alpha=0.5,
        )
    if experiment_data.problem_type == ProblemType.CLASSIFICATION:
        ax.set_ylim([0, 1])
    ax.set_xlim([min(experiment_data.full.x), max(experiment_data.full.x)])
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


def plot_1d_pls_prediction(
    experiment_data: ExperimentData,
    inducing_points: Data,
    x: torch.Tensor,
    predicted_samples: torch.Tensor,
    save_path: str,
    predicted_distribution: Optional[torch.distributions.Distribution] = None,
    title: str = None,
    is_sample_untransformed: bool = False,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    if not is_sample_untransformed:
        fig, ax = plot_1d_experiment_data(
            fig=fig,
            ax=ax,
            experiment_data=experiment_data,
            is_sample_untransformed=is_sample_untransformed,
        )
    if not is_sample_untransformed:
        ax.autoscale(enable=False)  # turn off autoscale before plotting particles
    if predicted_distribution:
        if isinstance(
            predicted_distribution, gpytorch.distributions.MultivariateNormal
        ):
            fig, ax = plot_1d_gp_prediction(
                fig=fig,
                ax=ax,
                x=experiment_data.full.x,
                mean=predicted_distribution.mean.detach(),
                variance=predicted_distribution.variance.detach(),
            )
        elif isinstance(predicted_distribution, torch.distributions.Bernoulli):
            fig, ax = plot_1d_gp_prediction(
                fig=fig,
                ax=ax,
                x=experiment_data.full.x,
                mean=predicted_distribution.probs.detach(),
                variance=None,
            )
        elif isinstance(predicted_distribution, torch.distributions.Poisson):
            fig, ax = plot_1d_gp_prediction(
                fig=fig,
                ax=ax,
                x=experiment_data.full.x,
                mean=predicted_distribution.rate.detach(),
                variance=None,
            )
        else:
            raise TypeError
    else:
        fig, ax = plot_1d_gp_prediction(
            fig=fig,
            ax=ax,
            x=experiment_data.full.x,
            mean=predicted_samples.mean(dim=1),
            variance=None,
        )
    for i in range(inducing_points.x.shape[0]):
        plt.axvline(
            x=inducing_points.x[i],
            color="tab:blue",
            alpha=0.2,
            label="induce" if i == 0 else None,
            zorder=0,
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


def plot_1d_pls_prediction_histogram(
    experiment_data: ExperimentData,
    predicted_samples: torch.Tensor,
    save_path: str,
    title: str = None,
    number_of_bins: int = 50,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    max_train_idx = torch.argmax(experiment_data.train.y)
    max_full_idx = torch.where(
        experiment_data.full.x == experiment_data.train.x[max_train_idx]
    )[0]
    histogram_data = predicted_samples[max_full_idx, :]
    ax.hist(histogram_data, bins=number_of_bins)
    ax.set_xlabel("y")
    ax.set_ylabel("count")
    if title is not None:
        ax.set_title(f"{title} (x={experiment_data.train.x[max_train_idx].item():.2f})")
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


def plot_1d_gp_prediction_and_inducing_points(
    model: Union[ExactGP, svGP],
    experiment_data: ExperimentData,
    title: str,
    save_path: str,
    inducing_points: Data = None,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    for i in range(inducing_points.x.shape[0]):
        plt.axvline(
            x=inducing_points.x[i],
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
    for i, step_size in enumerate(
        sorted(energy_potentials_history.keys(), reverse=True)
    ):
        shade = 0.1 + (i / (max(len(energy_potentials_history) - 1, 1))) * 0.8
        ax.plot(
            step_size * np.arange(len(energy_potentials_history[step_size])),
            np.log(energy_potentials_history[step_size]),
            label=step_size,
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


def animate_1d_pls_predictions(
    pls: PLSBase,
    number_of_particles: int,
    initial_particles_noise_only: bool,
    seed: int,
    step_size: float,
    number_of_epochs: int,
    experiment_data: ExperimentData,
    inducing_points: Data,
    x: torch.Tensor,
    title: str,
    save_path: str,
    christmas_colours: bool = False,
    animation_duration: int = 15,
    fps=15,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    plt.xlim(x.min(), x.max())
    ax.autoscale(enable=False)  # turn off autoscale before plotting particles

    particles = pls.initialise_particles(
        number_of_particles=number_of_particles,
        seed=seed,
        noise_only=initial_particles_noise_only,
    )
    predictive_noise = pls.sample_predictive_noise(
        x=experiment_data.full.x,
        particles=particles,
    ).detach()
    observation_noise = pls.sample_observation_noise(
        number_of_particles=number_of_particles,
        seed=seed,
    ).detach()
    predicted_samples = pls.predict_samples(
        x=experiment_data.full.x,
        particles=particles,
        predictive_noise=predictive_noise,
        observation_noise=observation_noise,
    ).detach()
    samples_plotted = [
        ax.plot(
            x.reshape(-1),
            predicted_samples[:, i].reshape(-1),
            color=[0.1, 0.1, 0.1]
            if not christmas_colours
            else ["green", "red", "blue"][i % 3],
            alpha=0.15,
            zorder=0,
            label="particle" if i == 0 else None,
        )[0]
        for i in range(predicted_samples.shape[-1])
    ]
    for i in range(inducing_points.x.shape[0]):
        plt.axvline(
            x=inducing_points.x[i],
            color="tab:blue",
            alpha=0.2,
            label="induce" if i == 0 else None,
            zorder=0,
        )
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
    number_of_frames = 1 + number_of_epochs // (
        (number_of_epochs // (animation_duration * fps)) + 1
    )
    progress_bar = tqdm(total=number_of_frames + 1, desc="PLS Particles GIF")

    def animate(iteration: int):
        for _ in range((number_of_epochs // (animation_duration * fps)) + 1):
            particle_wrapper.update(
                pls.calculate_particle_update(
                    particles=particle_wrapper.particles,
                    step_size=step_size,
                )
            )
        _predicted_samples = pls.predict_samples(
            x=experiment_data.full.x,
            particles=particle_wrapper.particles,
            predictive_noise=predictive_noise,
            observation_noise=observation_noise,
        ).detach()
        for i in range(_predicted_samples.shape[-1]):
            samples_plotted[i].set_data((x, _predicted_samples[:, i].reshape(-1)))
        ax.set_title(
            f"{title} (simulation time={step_size * particle_wrapper.num_updates:.2e})"
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


def animate_1d_pls_untransformed_predictions(
    pls: PLSBase,
    number_of_particles: int,
    initial_particles_noise_only: bool,
    seed: int,
    step_size: float,
    number_of_epochs: int,
    experiment_data: ExperimentData,
    inducing_points: Data,
    x: torch.Tensor,
    title: str,
    save_path: str,
    christmas_colours: bool = False,
    animation_duration: int = 15,
    fps=15,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    plt.xlim(x.min(), x.max())
    ax.autoscale(enable=False, axis="x")  # turn off autoscale before plotting particles
    particles = pls.initialise_particles(
        number_of_particles=number_of_particles,
        seed=seed,
        noise_only=initial_particles_noise_only,
    )
    predicted_untransformed_samples = pls.predict_untransformed_samples(
        x=experiment_data.full.x,
        particles=particles,
    ).detach()
    samples_plotted = [
        ax.plot(
            x.reshape(-1),
            predicted_untransformed_samples[:, i].reshape(-1),
            color=[0.1, 0.1, 0.1]
            if not christmas_colours
            else ["green", "red", "blue"][i % 3],
            alpha=0.15,
            zorder=0,
            label="particle" if i == 0 else None,
        )[0]
        for i in range(predicted_untransformed_samples.shape[-1])
    ]
    for i in range(inducing_points.x.shape[0]):
        plt.axvline(
            x=inducing_points.x[i],
            color="tab:blue",
            alpha=0.2,
            label="induce" if i == 0 else None,
            zorder=0,
        )
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
    number_of_frames = 1 + number_of_epochs // (
        (number_of_epochs // (animation_duration * fps)) + 1
    )
    progress_bar = tqdm(
        total=number_of_frames + 1, desc="PLS Untransformed Particles GIF"
    )

    def animate(iteration: int):
        for _ in range((number_of_epochs // (animation_duration * fps)) + 1):
            particle_wrapper.update(
                pls.calculate_particle_update(
                    particles=particle_wrapper.particles,
                    step_size=step_size,
                )
            )
        _predicted_untransformed_samples = pls.predict_untransformed_samples(
            x=experiment_data.full.x,
            particles=particle_wrapper.particles,
        ).detach()
        for i in range(_predicted_untransformed_samples.shape[-1]):
            samples_plotted[i].set_data(
                (x, _predicted_untransformed_samples[:, i].reshape(-1))
            )
        ax.set_title(
            f"{title} (simulation time={step_size * particle_wrapper.num_updates:.2e})"
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
    inducing_points: Data,
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
    fps=15,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    plt.xlim(experiment_data.full.x.min(), experiment_data.full.x.max())
    ax.autoscale(enable=False)  # turn off autoscale before plotting particles

    set_seed(seed)
    model = svGP(
        mean=mean,
        kernel=kernel,
        x_induce=inducing_points.x,
        likelihood=likelihood,
        learn_inducing_locations=learn_inducing_locations,
    )
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    all_params = set(model.parameters())

    if not learn_kernel_parameters:
        if isinstance(model.kernel, PLSKernel):
            all_params -= {model.kernel.base_kernel.base_kernel.raw_lengthscale}
            all_params -= {model.kernel.base_kernel.raw_outputscale}
        else:
            all_params -= {model.kernel.base_kernel.raw_lengthscale}
            all_params -= {model.kernel.raw_outputscale}
    model.likelihood.train()
    if torch.cuda.is_available():
        model.likelihood = model.likelihood.cuda()
    model.likelihood.train()

    optimizer = torch.optim.SGD(
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
            facecolor=(0.9, 0.9, 0.9),
            label="error bound (95%)",
            zorder=0,
        )
        mean_line = ax.plot(
            experiment_data.full.x.reshape(-1),
            mean_prediction.reshape(-1),
            label="mean",
            color="black",
            zorder=0,
        )[0]
    elif isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood):
        mean_line = ax.plot(
            experiment_data.full.x.reshape(-1),
            mean_prediction.reshape(-1),
            label="mean",
            zorder=0,
            color="black",
        )[0]
    else:
        raise NotImplementedError

    for i in range(inducing_points.x.shape[0]):
        plt.axvline(
            x=inducing_points.x[i],
            color="tab:blue",
            alpha=0.2,
            label="induce" if i == 0 else None,
            zorder=0,
        )
    plt.legend(loc="lower left")

    class Counter:
        def __init__(self):
            self.count = 0

        def increment(self):
            self.count += 1

    counter = Counter()
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
        ax.set_title(f"{title} (simulation time={learning_rate*counter.count:.2e})")
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
