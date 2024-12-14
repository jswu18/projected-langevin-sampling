from typing import List, Tuple, Union

import gpytorch
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.data import Data
from experiments.early_stopper import EarlyStopper
from src.gps import ExactGP, svGP
from src.projected_langevin_sampling import PLS, PLSKernel
from src.utils import set_seed


def train_exact_gp(
    data: Data,
    mean: gpytorch.means.Mean,
    kernel: gpytorch.kernels.Kernel,
    seed: int,
    number_of_epochs: int,
    learning_rate: float,
    likelihood: gpytorch.likelihoods.Likelihood,
    early_stopper_patience: float,
    model_name: str | None = None,
) -> Tuple[ExactGP, List[float]]:
    model_name = model_name if model_name is not None else "Exact GP"
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
    epochs_iter = tqdm(range(number_of_epochs), desc=f"{model_name} epoch")
    losses = []
    early_stopper = EarlyStopper(patience=early_stopper_patience)
    for _ in epochs_iter:
        optimizer.zero_grad()
        loss = -mll(model(data.x), data.y).sum()
        if early_stopper.should_stop(loss=loss.item(), step_size=learning_rate):
            break
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    return model, losses


def train_svgp(
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
    likelihood_noise: float | None = None,
) -> Tuple[svGP | None, List[float] | None]:
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
        if isinstance(
            likelihood, gpytorch.likelihoods.GaussianLikelihood
        ) or isinstance(likelihood, gpytorch.likelihoods.BernoulliLikelihood):
            model.likelihood.noise_covar.noise.data.fill_(likelihood_noise)
        if isinstance(likelihood, gpytorch.likelihoods.StudentTLikelihood):
            model.likelihood.noise.data.fill_(likelihood_noise)
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
    if torch.cuda.is_available():
        model = model.cuda()
        mll = mll.cuda()

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


def train_pls(
    pls: PLS,
    particles: torch.Tensor,
    number_of_epochs: int,
    step_size: float,
    early_stopper_patience: float,
    tqdm_desc: str | None = None,
) -> Tuple[torch.Tensor, List[float]]:
    energy_potentials = []
    early_stopper = EarlyStopper(patience=early_stopper_patience)
    for _ in tqdm(
        range(number_of_epochs),
        desc=tqdm_desc,
    ):
        particle_update = pls.calculate_particle_update(
            particles=particles,
            step_size=step_size,
        )
        particles += particle_update
        energy_potential = pls.calculate_energy_potential(particles=particles)
        if early_stopper.should_stop(loss=energy_potential, step_size=step_size):
            break
        energy_potentials.append(energy_potential)
    return particles, energy_potentials
