from typing import Dict, List, Tuple

import gpytorch
import torch
from matplotlib import pyplot as plt

from experiments.data import Data, ExperimentData

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
    model: gpytorch.models.GP,
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
    prediction = model(experiment_data.full.x)
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
