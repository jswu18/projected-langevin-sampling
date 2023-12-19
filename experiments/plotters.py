from typing import Dict, List, Tuple, Union

import gpytorch
import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from experiments.data import Data, ExperimentData
from src.gps import ExactGP, svGP
from src.gradient_flows import ProjectedWassersteinGradientFlow
from src.utils import set_seed

_DATA_COLORS = {
    "train": "tab:blue",
    "validation": "tab:green",
    "test": "tab:orange",
    "induce": "black",
}

_DATA_TRANSPARENCY = {
    "train": 0.3,
    "validation": 0.3,
    "test": 0.3,
    "induce": 1.0,
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
        facecolor=(0.8, 0.8, 0.8),
        label="error bound (95%)",
        zorder=0,
    )
    ax.plot(x.reshape(-1), mean.reshape(-1), label="mean", zorder=0)
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
) -> Tuple[plt.Figure, plt.Axes]:
    if data.name in _DATA_COLORS:
        color = _DATA_COLORS[data.name]
    else:
        color = "tab:blue"
    if data.name in _DATA_TRANSPARENCY:
        alpha = _DATA_TRANSPARENCY[data.name]
    else:
        alpha = 1.0
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
            )
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
        color=[0.3, 0.3, 0.3],
        alpha=0.1,
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
    title: str = None,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    fig, ax = plot_1d_data(
        fig=fig,
        ax=ax,
        data=induce_data,
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
    losses_history: List[List[float]],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for i, losses in enumerate(losses_history):
        shade = ((len(losses_history) - i) / len(losses_history)) * 0.8
        ax.plot(losses, label=f"iteration {i + 1}", color=[shade, shade, shade])
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
    if induce_data is not None:
        fig, ax = plot_1d_data(
            fig=fig,
            ax=ax,
            data=induce_data,
        )
    prediction = model.likelihood(model(experiment_data.full.x))
    fig, ax = plot_1d_gp_prediction(
        fig=fig,
        ax=ax,
        x=experiment_data.full.x,
        mean=prediction.mean.detach(),
        variance=prediction.variance.detach(),
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_update_magnitude(
    update_log_magnitude_history: Dict[float, List[float]],
    title: str,
    save_path: str,
):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for name, update_magnitudes in update_log_magnitude_history.items():
        ax.plot(update_magnitudes, label=name)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("update log magnitude")
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
    pwgf: ProjectedWassersteinGradientFlow,
    seed: int,
    learning_rate: float,
    number_of_epochs: int,
    experiment_data: ExperimentData,
    induce_data: Data,
    x: torch.Tensor,
    title: str,
    save_path: str,
    christmas_colours: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    fig, ax = plot_1d_data(
        fig=fig,
        ax=ax,
        data=induce_data,
    )
    plt.xlim(x.min(), x.max())
    plt.ylim(
        experiment_data.full.y.min() - 1,
        experiment_data.full.y.max() + 1,
    )

    pwgf.reset_particles(seed=seed)
    predicted_samples = pwgf.predict_samples(
        x=experiment_data.full.x,
        include_observation_noise=False,
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

    def animate(iteration: int):
        _ = pwgf.update(
            learning_rate=torch.tensor(learning_rate),
        )
        _predicted_samples = pwgf.predict_samples(
            x=experiment_data.full.x,
            include_observation_noise=False,
        ).detach()
        for i in range(_predicted_samples.shape[-1]):
            samples_plotted[i].set_data((x, _predicted_samples[:, i].reshape(-1)))
        ax.set_title(f"{title} ({iteration=})")
        return (samples_plotted[0],)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=number_of_epochs, interval=50
    )

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(
        fps=15,
        bitrate=1800,
    )
    ani.save(save_path, writer=writer)


def animate_1d_gp_predictions(
    experiment_data: ExperimentData,
    induce_data: Data,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    seed: int,
    number_of_epochs: int,
    batch_size: int,
    learning_rate: float,
    title: str,
    save_path: str,
    learn_inducing_locations: bool,
    learn_kernel_parameters: bool,
    learn_observation_noise: bool,
    observation_noise: float = None,
    christmas_colours: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig, ax = plot_1d_experiment_data(
        fig=fig,
        ax=ax,
        experiment_data=experiment_data,
    )
    fig, ax = plot_1d_data(
        fig=fig,
        ax=ax,
        data=induce_data,
    )
    plt.xlim(experiment_data.full.x.min(), experiment_data.full.x.max())
    plt.ylim(
        experiment_data.full.y.min() - 1,
        experiment_data.full.y.max() + 1,
    )

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
        model.likelihood, model, num_data=experiment_data.train.x.shape[0]
    )

    train_dataset = TensorDataset(experiment_data.train.x, experiment_data.train.y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    prediction = model.likelihood(model(experiment_data.full.x))
    mean_prediction = prediction.mean.detach()
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
    plt.legend(loc="lower left")

    def animate(iteration: int):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
        _prediction = model.likelihood(model(experiment_data.full.x))
        _mean_prediction = _prediction.mean.detach()
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
        ax.set_title(f"{title} ({iteration=})")
        if christmas_colours:
            fill.set_color(
                (
                    0.5 * np.sin(iteration / 15) + 0.5,
                    0.5 * np.sin(iteration / 15 + 2 * np.pi / 3) + 0.5,
                    0.5 * np.sin(iteration / 15 + 4 * np.pi / 3) + 0.5,
                )
            )
        print(iteration)
        return (mean_line,)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=number_of_epochs, interval=50
    )

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(
        fps=30,
        bitrate=1800,
    )
    ani.save(save_path, writer=writer)
