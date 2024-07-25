from typing import Dict, List, Tuple, Union

import gpytorch
import matplotlib.animation as animation
import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.data import Data, ExperimentData, ProblemType
from src.conformalise import ConformaliseGP
from src.conformalise.base import ConformaliseBase, ConformalPrediction
from src.distributions import StudentTMarginals
from src.gps import ExactGP, svGP
from src.kernels import PLSKernel
from src.projected_langevin_sampling import ProjectedLangevinSampling
from src.projected_langevin_sampling.basis import OrthonormalBasis
from src.utils import set_seed

_DATA_COLORS = {
    "train": "tab:blue",
    "test": "tab:orange",
}

_DATA_TRANSPARENCY = {
    "train": 0.5,
    "test": 0.5,
}


def plot_1d_gp_prediction(
    fig: plt.Figure,
    ax: plt.Axes,
    x: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor | None = None,
    coverage: float = 0.95,
    save_path: str | None = None,
    title: str | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    x = x.cpu().detach()
    mean = mean.cpu().detach()
    variance = variance.cpu().detach() if variance is not None else None
    if variance is not None:
        stdev = torch.sqrt(variance)
        confidence_interval_scale = scipy.stats.norm.interval(coverage)[1]
        ax.fill_between(
            x.reshape(-1),
            (mean - confidence_interval_scale * stdev).reshape(-1),
            (mean + confidence_interval_scale * stdev).reshape(-1),
            facecolor=(0.9, 0.9, 0.9),
            label=f"{coverage*100}% error",
            zorder=0,
        )
    ax.plot(
        x.reshape(-1),
        mean.reshape(-1),
        label="mean",
        zorder=1,
        color="black",
        linewidth=0.5,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.legend(
            loc="outside lower center",
            ncols=3,
        )
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return fig, ax
    else:
        return fig, ax


def plot_1d_data(
    fig: plt.Figure,
    ax: plt.Axes,
    data: Data,
    save_path: str | None = None,
    title: str | None = None,
    color: str = "tab:blue",
    alpha: float = 0.3,
    s: int = 10,
) -> Tuple[plt.Figure, plt.Axes]:
    if data.name in _DATA_COLORS:
        color = _DATA_COLORS[data.name]
    if data.name in _DATA_TRANSPARENCY:
        alpha = _DATA_TRANSPARENCY[data.name]
    if data.y is not None:
        ax.scatter(
            data.x.cpu(),
            data.y.cpu(),
            label=data.name,
            alpha=alpha,
            color=color,
            s=s,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.legend(
            loc="outside lower center",
            ncols=3,
        )
        fig.savefig(save_path)
        plt.close(fig)
        return fig, ax
    else:
        return fig, ax


def plot_1d_experiment_data(
    fig: plt.Figure,
    ax: plt.Axes,
    experiment_data: ExperimentData,
    save_path: str | None = None,
    title: str | None = None,
    is_sample_untransformed: bool = False,
    alpha: float = 0.3,
    s: int = 10,
) -> Tuple[plt.Figure, plt.Axes]:
    for data in [
        experiment_data.train,
        experiment_data.validation,
        experiment_data.test,
    ]:
        if data is None:
            continue
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
            alpha=alpha,
            s=s,
        )
    if not is_sample_untransformed and experiment_data.full.y_untransformed is not None:
        ax.plot(
            experiment_data.full.x.cpu(),
            experiment_data.full.y_untransformed.reshape(
                experiment_data.full.x.shape
            ).cpu(),
            label="latent",
            color="midnightblue",
            linestyle=(0, (3, 1, 1, 1, 1, 1)),
            linewidth=1,
        )
    if experiment_data.problem_type == ProblemType.CLASSIFICATION:
        ax.set_ylim([0, 1])
    if experiment_data.problem_type == ProblemType.POISSON_REGRESSION:
        ax.set_ylim(bottom=0)
    ax.set_xlim([min(experiment_data.full.x.cpu()), max(experiment_data.full.x.cpu())])
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
    alpha: float = 0.2,
) -> Tuple[plt.Figure, plt.Axes]:
    ax.plot(
        x.cpu().reshape(-1),
        y.cpu().reshape(-1),
        color="black",
        alpha=alpha,
        zorder=0,
        label="particle" if add_label else None,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def plot_1d_pls_prediction(
    experiment_data: ExperimentData,
    x: torch.Tensor,
    save_path: str,
    coverage: float = 0.95,
    predicted_samples: torch.Tensor | None = None,
    y_name: str | None = None,
    predicted_distribution: torch.distributions.Distribution | None = None,
    title: str | None = None,
    is_sample_untransformed: bool = False,
    max_particles_to_plot: int = 50,
):
    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
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
                coverage=coverage,
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
                coverage=coverage,
            )
        elif isinstance(predicted_distribution, StudentTMarginals):
            fig, ax = plot_1d_gp_prediction(
                fig=fig,
                ax=ax,
                x=experiment_data.full.x,
                mean=predicted_distribution.loc,
                variance=predicted_distribution.scale**2,
                coverage=coverage,
            )
        else:
            raise TypeError
    if predicted_samples is not None:
        for i in range(min(predicted_samples.shape[1], max_particles_to_plot)):
            fig, ax = plot_1d_particle(
                fig=fig,
                ax=ax,
                x=x,
                y=predicted_samples[:, i],
                add_label=i == 0,
            )
    if y_name is not None:
        ax.set_ylabel(y_name)
    if title is not None:
        ax.set_title(title)
    fig.legend(
        loc="outside lower center",
        ncols=3,
    )
    fig.savefig(
        save_path,
    )
    plt.close(fig)


def plot_1d_pls_prediction_histogram(
    experiment_data: ExperimentData,
    x: torch.Tensor,
    predicted_samples: torch.Tensor,
    untransformed_predicted_samples: torch.Tensor,
    save_path: str,
    title: str | None = None,
    number_of_bins: int = 50,
):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3), layout="constrained")
    for i in range(predicted_samples.shape[1]):
        fig, ax[0] = plot_1d_particle(
            fig=fig,
            ax=ax[0],
            x=x,
            y=untransformed_predicted_samples[:, i],
            add_label=i == 0,
            alpha=0.1,
        )
    if title is not None:
        ax[0].set_title(f"$f(x)$")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$f(x)$")
    ax[0].set_xlim([min(x).cpu(), max(x).cpu()])
    max_train_idx = torch.argmax(experiment_data.train.y)
    max_full_idx = torch.where(
        experiment_data.full.x == experiment_data.train.x[max_train_idx]
    )[0]

    ax[0].axvline(
        x=experiment_data.train.x[max_train_idx].item(),
        color="tab:red",
        label="cross section",
        linewidth=3,
        alpha=0.75,
    )

    histogram_data = untransformed_predicted_samples[max_full_idx, :]
    ax[1].hist(
        histogram_data.cpu(),
        bins=number_of_bins,
        color="tab:red",
        alpha=0.75,
    )
    ax[1].set_xlabel(f"$f(x)$ bin")
    ax[1].set_ylabel("count")
    if title is not None:
        ax[1].set_title(
            f"Histogram at $f(x={experiment_data.train.x[max_train_idx].item():.2f})$"
        )
    fig.legend(loc="outside lower center", ncols=2)

    fig.savefig(
        save_path,
    )
    plt.close(fig)


def plot_losses(
    losses_history: Dict[float, List[float]],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
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
        fig.legend(
            loc="outside lower center",
            ncols=3,
        )

    fig.savefig(
        save_path,
    )
    plt.close(fig)


def plot_1d_gp_prediction_and_inducing_points(
    model: Union[ExactGP, svGP],
    experiment_data: ExperimentData,
    title: str,
    save_path: str,
    inducing_points: Data | None = None,
    coverage: float = 0.95,
):
    model.eval()
    model.likelihood.eval()
    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    if inducing_points is not None:
        for i in range(inducing_points.x.shape[0]):
            plt.axvline(
                x=inducing_points.x[i].cpu(),
                color="black",
                alpha=0.2,
                label="induce" if i == 0 else None,
                zorder=1,
            )
    ax.autoscale(enable=False)  # turn off autoscale before plotting gp prediction
    prediction = model.likelihood(model(experiment_data.full.x))
    if isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood):
        fig, ax = plot_1d_gp_prediction(
            fig=fig,
            ax=ax,
            x=experiment_data.full.x.cpu(),
            mean=prediction.mean.detach(),
            variance=prediction.variance.detach(),
            coverage=coverage,
        )
    elif isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood):
        ax.plot(
            experiment_data.full.x.reshape(-1).cpu(),
            prediction.mean.detach().reshape(-1).cpu(),
            label="prediction",
            zorder=0,
            color="black",
        )
    elif isinstance(model.likelihood, gpytorch.likelihoods.StudentTLikelihood):
        fig, ax = plot_1d_gp_prediction(
            fig=fig,
            ax=ax,
            x=experiment_data.full.x.reshape(-1).cpu(),
            mean=prediction.mean.mean(axis=0).detach().reshape(-1).cpu()
            if prediction.mean.dim() == 2
            else prediction.mean.detach().reshape(-1).cpu(),
            variance=prediction.variance.mean(axis=0).detach(),
            coverage=coverage,
        )
    else:
        raise NotImplementedError
    fig.legend(
        loc="outside lower center",
        ncols=3,
    )
    ax.set_title(title)

    fig.savefig(save_path)
    plt.close(fig)


def plot_1d_conformal_prediction(
    model: ConformaliseBase,
    experiment_data: ExperimentData,
    plot_title: str,
    save_path: str,
    coverage: float,
    inducing_points: Data | None = None,
):
    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    if inducing_points is not None:
        for i in range(inducing_points.x.shape[0]):
            plt.axvline(
                x=float(inducing_points.x[i].cpu()),
                color="black",
                alpha=0.2,
                label="induce" if i == 0 else None,
                zorder=1,
            )
    ax.autoscale(enable=False)  # turn off autoscale before plotting gp prediction
    prediction = model.predict(x=experiment_data.full.x, coverage=coverage)
    ax.fill_between(
        experiment_data.full.x.reshape(-1).cpu().detach().numpy(),
        prediction.lower.reshape(-1).cpu().detach().numpy(),
        prediction.upper.reshape(-1).cpu().detach().numpy(),
        facecolor=(0.9, 0.9, 0.9),
        label=f"{coverage*100}% error",
        zorder=0,
    )
    ax.plot(
        experiment_data.full.x.reshape(-1).cpu().detach().numpy(),
        prediction.mean.reshape(-1).cpu().detach().numpy(),
        label="mean",
        zorder=1,
        color="black",
        linewidth=0.5,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if plot_title is not None:
        ax.set_title(plot_title)
    fig.legend(
        loc="outside lower center",
        ncols=3,
    )
    fig.savefig(save_path)
    plt.close(fig)


def plot_energy_potentials(
    energy_potentials_history: Dict[float, List[float]],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(13, 6), layout="constrained")
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
    fig.legend(
        loc="outside lower center",
        ncols=3,
    )

    fig.savefig(
        save_path,
    )
    plt.close(fig)


def plot_true_versus_predicted(
    y_true: torch.Tensor,
    y_pred: gpytorch.distributions.MultivariateNormal
    | ConformalPrediction
    | StudentTMarginals,
    title: str,
    save_path: str,
    coverage: float,
    error_bar: bool = True,
):
    fig, ax = plt.subplots(figsize=(13, 13), layout="constrained")
    y_pred_mean = y_pred.loc if isinstance(y_pred, StudentTMarginals) else y_pred.mean
    if error_bar:
        if isinstance(y_pred, gpytorch.distributions.MultivariateNormal):
            confidence_interval_scale = scipy.stats.norm.interval(coverage)[1]
            _, _, bars = plt.errorbar(
                y_true.cpu().detach().numpy(),
                y_pred_mean.cpu().detach().numpy(),
                yerr=confidence_interval_scale * y_pred.variance.cpu().detach().numpy(),
                fmt="o",
                elinewidth=0.5,
            )
            [bar.set_alpha(0.5) for bar in bars]
        elif isinstance(y_pred, ConformalPrediction):
            assert coverage == y_pred.coverage, f"{coverage=}!={y_pred.coverage=}"
            _, _, bars = plt.errorbar(
                y_true.cpu().detach().numpy(),
                y_pred_mean.cpu().detach().numpy(),
                yerr=[
                    (y_pred.mean - y_pred.lower).cpu().detach().numpy().clip(0, None),
                    (y_pred.upper - y_pred.mean).cpu().detach().numpy().clip(0, None),
                ],
                fmt="o",
                elinewidth=0.5,
            )
        [bar.set_alpha(0.5) for bar in bars]
    else:
        plt.scatter(
            y_true.cpu().detach().numpy(),
            y_pred_mean.cpu().detach().numpy(),
            marker="o",
        )
    ax.set_xlabel("true")
    ax.set_ylabel("predicted")
    if error_bar:
        ax.set_title(f"{title} ({coverage*100}% confidence interval)")
    else:
        ax.set_title(title)
    axis_lims = [
        min(y_true.min().cpu().item(), y_pred_mean.min().cpu().item()),
        max(y_true.max().cpu().item(), y_pred_mean.max().cpu().item()),
    ]
    plt.plot(axis_lims, axis_lims, color="gray", label="target", linestyle="dashed")
    plt.xlim(axis_lims)
    plt.ylim(axis_lims)
    fig.legend(
        loc="outside lower center",
        ncols=3,
    )

    fig.savefig(
        save_path,
    )
    plt.close(fig)


def animate_1d_pls_predictions(
    pls: ProjectedLangevinSampling,
    number_of_particles: int,
    initial_particles_noise_only: bool,
    seed: int,
    step_size: float,
    number_of_epochs: int,
    experiment_data: ExperimentData,
    x: torch.Tensor,
    title: str,
    save_path: str,
    christmas_colours: bool = False,
    animation_duration: int = 10,
    max_particles_to_plot: int = 50,
    fps: int = 15,
) -> None:
    number_of_particles = min(number_of_particles, max_particles_to_plot)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
        alpha=0.1,
        s=30,
    )
    plt.xlim(x.min().cpu(), x.max().cpu())
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
            x.reshape(-1).cpu(),
            predicted_samples[:, i].reshape(-1).cpu(),
            color="black" if not christmas_colours else ["green", "red", "blue"][i % 3],
            alpha=0.2,
            zorder=1,
            label="particle" if i == 0 else None,
        )[0]
        for i in range(predicted_samples.shape[-1])
    ]
    fig.legend(
        loc="outside lower center",
        ncols=4,
    )

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
    epochs_per_frame = (number_of_epochs // (animation_duration * fps)) + 1
    number_of_frames = number_of_epochs // epochs_per_frame
    progress_bar = tqdm(total=number_of_frames + 1, desc="PLS Particles GIF")

    def animate(iteration: int):
        for _ in range(epochs_per_frame):
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
            samples_plotted[i].set_data(
                (x.cpu(), _predicted_samples[:, i].reshape(-1).cpu())
            )
        ax.set_title(f"{title} (t={step_size * particle_wrapper.num_updates:.2e})")
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
    pls: ProjectedLangevinSampling,
    number_of_particles: int,
    initial_particles_noise_only: bool,
    seed: int,
    step_size: float,
    number_of_epochs: int,
    experiment_data: ExperimentData,
    x: torch.Tensor,
    title: str,
    save_path: str,
    christmas_colours: bool = False,
    animation_duration: int = 10,
    max_particles_to_plot: int = 50,
    fps: int = 15,
    number_of_bins: int = 50,
) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(6, 9), layout="constrained")
    fig, ax[0] = plot_1d_experiment_data(
        fig=fig,
        ax=ax[0],
        experiment_data=experiment_data,
        alpha=0.1,
        s=30,
    )
    ax[0].set_xlim(x.min().cpu(), x.max().cpu())
    ax[1].set_xlim(x.min().cpu(), x.max().cpu())
    ax[0].autoscale(
        enable=False, axis="x"
    )  # turn off autoscale before plotting particles
    ax[1].autoscale(
        enable=False, axis="x"
    )  # turn off autoscale before plotting particles
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
    transformed_samples_plotted = [
        ax[0].plot(
            x.reshape(-1).cpu(),
            predicted_samples[:, i].reshape(-1).cpu(),
            color="black" if not christmas_colours else ["green", "red", "blue"][i % 3],
            alpha=0.2,
            zorder=1,
            label="particle" if i == 0 else None,
        )[0]
        for i in range(min(max_particles_to_plot, predicted_samples.shape[-1]))
    ]
    predicted_untransformed_samples = pls.predict_untransformed_samples(
        x=experiment_data.full.x,
        particles=particles,
        noise=predictive_noise,
    ).detach()
    untransformed_samples_plotted = [
        ax[1].plot(
            x.reshape(-1).cpu(),
            predicted_untransformed_samples[:, i].reshape(-1).cpu(),
            color="black" if not christmas_colours else ["green", "red", "blue"][i % 3],
            alpha=0.2,
            zorder=1,
            label="particle" if i == 0 else None,
        )[0]
        for i in range(min(max_particles_to_plot, predicted_samples.shape[-1]))
    ]
    max_train_idx = torch.argmax(experiment_data.train.y)
    max_full_idx = torch.where(
        experiment_data.full.x == experiment_data.train.x[max_train_idx]
    )[0]
    ax[1].axvline(
        x=experiment_data.train.x[max_train_idx].cpu().item(),
        color="tab:red",
        label="cross section",
        linewidth=3,
        alpha=0.75,
    )
    histogram_data = predicted_untransformed_samples[max_full_idx, :]
    counts, bins = np.histogram(
        histogram_data.cpu(),
        bins=number_of_bins,
    )
    histogram = ax[2].stairs(
        counts,
        bins,
        color="tab:red",
        alpha=0.75,
        fill=True,
    )
    ax[2].set_ylim(0, 1.5 * max(counts))
    ax[2].autoscale(enable=False, axis="y")  # turn off autoscale
    ax[2].set_xlabel(f"$f(x)$ bin")
    ax[2].set_ylabel("count")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")

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
        _predicted_samples = pls.predict_samples(
            x=experiment_data.full.x,
            particles=particle_wrapper.particles,
            predictive_noise=predictive_noise,
            observation_noise=observation_noise,
        ).detach()
        for i in range(min(max_particles_to_plot, _predicted_samples.shape[-1])):
            transformed_samples_plotted[i].set_data(
                (x.cpu(), _predicted_samples[:, i].reshape(-1).cpu())
            )
        _predicted_untransformed_samples = pls.predict_untransformed_samples(
            x=experiment_data.full.x,
            particles=particle_wrapper.particles,
            noise=predictive_noise,
        ).detach()
        for i in range(
            min(max_particles_to_plot, _predicted_untransformed_samples.shape[-1])
        ):
            untransformed_samples_plotted[i].set_data(
                (x.cpu(), _predicted_untransformed_samples[:, i].reshape(-1).cpu())
            )
        _histogram_data = _predicted_untransformed_samples[max_full_idx, :]
        _counts, _ = np.histogram(
            _histogram_data.cpu(),
            bins=bins,
        )
        histogram.set_data(_counts, bins)
        ax[0].set_title(f"$f(x)^2$ (t={step_size * particle_wrapper.num_updates:.2e})")
        ax[1].set_title(f"$f(x)$ (t={step_size * particle_wrapper.num_updates:.2e})")
        ax[2].set_title(
            f"Histogram at $f(x={experiment_data.train.x[max_train_idx].item():.2f}) (t={step_size * particle_wrapper.num_updates:.2e})$"
        )
        progress_bar.update(n=1)
        return (transformed_samples_plotted[0],)

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
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.likelihoods.BernoulliLikelihood,
        gpytorch.likelihoods.StudentTLikelihood,
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
    animation_duration: int = 10,
    fps: int = 15,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
        alpha=0.1,
        s=30,
    )
    if not learn_inducing_locations:
        for i in range(inducing_points.x.shape[0]):
            plt.axvline(
                x=inducing_points.x[i].cpu(),
                color="black",
                alpha=0.2,
                label="induce" if i == 0 else None,
                zorder=1,
            )
    plt.xlim(experiment_data.full.x.min().cpu(), experiment_data.full.x.max().cpu())
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
    all_params = set(model.parameters())

    if not learn_kernel_parameters:
        if isinstance(model.kernel, PLSKernel):
            all_params -= {model.kernel.base_kernel.base_kernel.raw_lengthscale}
            all_params -= {model.kernel.base_kernel.raw_outputscale}
        else:
            all_params -= {model.kernel.base_kernel.raw_lengthscale}
            all_params -= {model.kernel.raw_outputscale}
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
    if torch.cuda.is_available():
        mll = mll.cuda()

    train_dataset = TensorDataset(experiment_data.train.x, experiment_data.train.y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    prediction = model.likelihood(model(experiment_data.full.x))
    mean_prediction = prediction.mean.detach()
    if isinstance(model.likelihood, gpytorch.likelihoods.GaussianLikelihood):
        stdev_prediction = torch.sqrt(prediction.variance.detach())
        fill = ax.fill_between(
            experiment_data.full.x.reshape(-1).cpu(),
            (mean_prediction - 1.96 * stdev_prediction).reshape(-1).cpu(),
            (mean_prediction + 1.96 * stdev_prediction).reshape(-1).cpu(),
            facecolor=(0.9, 0.9, 0.9),
            label="95% error",
            zorder=0,
        )
        mean_line = ax.plot(
            experiment_data.full.x.reshape(-1).cpu(),
            mean_prediction.reshape(-1).cpu(),
            label="mean",
            color="black",
            zorder=0,
        )[0]
    elif isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood):
        mean_line = ax.plot(
            experiment_data.full.x.reshape(-1).cpu(),
            mean_prediction.reshape(-1).cpu(),
            label="mean",
            zorder=0,
            color="black",
        )[0]
    else:
        raise NotImplementedError
    fig.legend(
        loc="outside lower center",
        ncols=3,
    )

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
                (_mean_prediction - 1.96 * _stdev_prediction).reshape(-1).cpu()
            )
            verts[experiment_data.full.x.shape[0] + 2 : -1, 1] = list(
                (_mean_prediction + 1.96 * _stdev_prediction).reshape(-1).cpu()
            )[::-1]
            mean_line.set_data(
                (
                    experiment_data.full.x.reshape(-1).cpu(),
                    _mean_prediction.reshape(-1).cpu(),
                )
            )
        elif isinstance(model.likelihood, gpytorch.likelihoods.BernoulliLikelihood):
            mean_line.set_data(
                (
                    experiment_data.full.x.reshape(-1).cpu(),
                    _mean_prediction.reshape(-1).cpu(),
                )
            )
        else:
            raise NotImplementedError
        ax.set_title(f"{title} (t={learning_rate*counter.count:.2e})")
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


def plot_eigenvalues(basis: OrthonormalBasis, save_path: str, title: str) -> None:
    eigenvalues, _ = torch.linalg.eigh(
        (1 / basis.x_induce.shape[0]) * basis.base_gram_induce.evaluate()
    )
    fig, ax = plt.subplots(figsize=(13, 13), layout="constrained")
    ax.bar(
        np.arange(1, eigenvalues.shape[0] + 1),
        np.flip(eigenvalues.cpu().detach().numpy()),
    )
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    ax.set_title(title)
    fig.savefig(save_path)
    plt.close()
