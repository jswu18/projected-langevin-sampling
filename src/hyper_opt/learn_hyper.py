import os
import sys
import warnings

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# Get the path to the directory containing 'src'
src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(src)

from costs import Costs, GaussianCost
from sklearn.model_selection import train_test_split

from hyper_opt.make_sample import train_x, train_x_all, train_y
from hyper_opt.utils import check_convergence, custom_dataloader
from utils import StandardScaler  # torch version of standardscaler
from utils import median_heuristic


class KernelsHyperOpt:
    def __init__(
        self,
        cost: Costs,
        kernel: gpytorch.kernels.Kernel,
        device,
        standardize=True,
        reg_para=0.001,
    ) -> None:
        self.cost = cost
        self.device = device
        self.alpha = (
            None  # initialised in data_loader, before we dont know length of alpha
        )
        self.beta = None
        self.reg_para = reg_para
        self.standardize = standardize
        self.Xscaler = None
        self.Yscaler = None
        self.dl1 = None
        self.X1 = None
        self.dl2 = None
        self.X2 = None
        self.optim1 = None
        self.optim2 = None
        self.optim_beta = None
        self.kernel = kernel

        if self.standardize:
            self.kernel.outputscale = torch.tensor(1.0)
            self.kernel.raw_outputscale.requires_grad_(False)

        self.kernel = self.kernel.to(device)

    def _init_optimization(self, X, Y, lr1, lr2, n_hyper, batch_size):
        # Split Data
        if self.standardize:
            Xscaler = StandardScaler()
            Yscaler = StandardScaler()
            X = Xscaler.fit_transform(X)
            Y = Yscaler.fit_transform(Y)
            self.Xscaler = Xscaler
            self.Yscaler = Yscaler

        X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=n_hyper)
        self.dl1 = custom_dataloader(X1, Y1, batch_size)
        self.dl2 = custom_dataloader(X2, Y2, batch_size)

        self.X1 = torch.tensor(X1, dtype=torch.float32).to(device)
        self.X2 = torch.tensor(X2, dtype=torch.float32).to(device)

        self.Y1 = torch.tensor(Y1, dtype=torch.float32).to(device)
        self.Y2 = torch.tensor(Y2, dtype=torch.float32).to(device)

        self.alpha = torch.nn.Parameter(
            torch.ones(self.X1.shape[0], device=self.device)
        )
        self.beta = torch.nn.Parameter(torch.ones(self.X1.shape[0], device=self.device))
        # Set requires_grad to False
        self.alpha.requires_grad = False

        # Initialize optimizer with parameters
        self.optim1 = optim.SGD([self.alpha], lr=lr1)
        self.optim_beta = optim.Adam([self.beta], lr=lr1)
        self.kernel.base_kernel.lengthscale = median_heuristic(X.unsqueeze(1))
        # self.kernel.base_kernel.lengthscale = torch.tensor(1.0)

        all_parameters = list(kernel.parameters())
        trainable_parameters = [
            param for param in all_parameters if param.requires_grad
        ]
        self.optim2 = optim.Adam(trainable_parameters, lr=lr2)

    def _update_alpha(self, max_epochs=10000):
        N1 = self.X1.shape[0]
        cost_list = []

        for epoch in range(max_epochs):
            # Train One Epoche
            cost_epoch = 0
            for X_batch, Y_batch in self.dl1:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                # Calculate gradient
                k_Xb_X1 = self.kernel(X_batch, self.X1).evaluate()
                Y_batch_hat = torch.matmul(k_Xb_X1, self.alpha)

                N_batch = Y_batch.shape[0]
                grad = (
                    1
                    / N_batch
                    * torch.matmul(
                        k_Xb_X1.T, self.cost.derivative(Y_batch, Y_batch_hat)
                    )
                    + self.reg_para * self.alpha
                )

                # Precondition, That didnt work at all. Hail Adam
                # grad_precond = torch.linalg.solve_triangular(self.cholesky1, grad.unsqueeze(1), upper=False)
                # self.alpha.grad =  grad_precond.squeeze(1)

                self.alpha.grad = grad

                self.optim1.step()
                self.optim1.zero_grad()

                with torch.no_grad():
                    cost_epoch += 1 / N_batch * self.cost(
                        Y_batch, Y_batch_hat
                    ) + self.reg_para * torch.sum(self.alpha**2)

            cost_list.append(cost_epoch / len(self.dl1))
            print(
                f"Epoch {epoch+1}, Average Costs Alpha Optimisation: {cost_list[-1]:.2f}"
            )

            if epoch > self.patience:
                if check_convergence(
                    cost_list, patience=self.patience, min_delta=self.min_delta
                ):
                    break

    def _update_beta(self, max_epochs=10000):
        N1 = self.X1.shape[0]
        cost_list = []

        for epoch in range(max_epochs):
            loss = 1 / N1 * self.cost(self.Y1, self.beta)
            loss.backward()
            self.optim_beta.step()
            self.optim_beta.zero_grad()

            with torch.no_grad():
                cost_list.append(loss)
                print(
                    f"Epoch {epoch+1}, Average Costs Beta Optimisation: {cost_list[-1]:.2f}"
                )

            if epoch > self.patience:
                if check_convergence(
                    cost_list, patience=self.patience, min_delta=self.min_delta
                ):
                    # Update Alpha

                    break

    def _update_kernel_params(self, max_epochs=100):
        cost_list = []
        N2 = self.X2.shape[0]

        has_converged = False
        for epoch in range(max_epochs):
            cost_epoch = 0
            # Train One Epoche
            for X_batch, Y_batch in self.dl2:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                # Calculate gradient
                k_Xb_X1 = self.kernel(X_batch, self.X1).evaluate()
                Y_batch_hat = torch.matmul(k_Xb_X1, self.alpha)

                N_batch = X_batch.shape[0]
                loss = 1 / N_batch * self.cost(Y_batch, Y_batch_hat)

                # Backward pass and optimization
                self.optim2.zero_grad()
                loss.backward()
                self.optim2.step()

                with torch.no_grad():
                    cost_epoch += loss

            cost_list.append(cost_epoch / len(self.dl2))
            print(
                f"Epoch {epoch+1}, Average Costs Hyerparameter Optimisation: {cost_list[-1]:.2f}"
            )

            if epoch > self.patience:
                if check_convergence(
                    cost_list, patience=self.patience, min_delta=self.min_delta
                ):
                    has_converged = True
                    return has_converged

        return has_converged

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size,
        max_iter=100,
        lr1=1e-4,
        lr2=1e-4,
        n_hyper=0.2,
        patience=5,
        min_delta=0.001,
    ):
        # Initialise Dataloaders, optimisers, etc
        self._init_optimization(
            X, Y, lr1=lr1, lr2=lr2, n_hyper=n_hyper, batch_size=batch_size
        )

        # Initialise other hyperparamters
        self.patience = patience
        self.min_delta = min_delta

        counter = 0
        k_XX = self.kernel(self.X1, self.X1) + self.reg_para * torch.eye(
            self.X1.shape[0]
        )

        while counter < max_iter:
            # self._update_alpha()
            self._update_beta()
            with torch.no_grad():
                self.alpha = torch.linalg.solve(k_XX, self.beta)

            has_converged = self._update_kernel_params()

            if has_converged:
                print("The Algorithm has stopped due to convergence.")
                break

            counter += 1

        if counter == max_iter:
            warnings.warn(
                "The Algorithm has stopped due to reaching the maximum number of iterations. ",
                category=UserWarning,
            )

        # Access and print the transformed parameters
        print("Transformed Kernel Parameters:")
        for name, param in kernel.named_parameters():
            # Check if the parameter has a transformation
            if hasattr(param, "transform"):
                # Get the transformed value
                transformed_value = param.transform(param).item()
                print(f"Parameter: {name}")
                print(f"Transformed Value: {transformed_value}")
                print("-------")
            else:
                # Directly access if no transformation is applied
                print(f"Parameter: {name}")
                print(f"Value: {param.item()}")
                print("-------")

    def predict(self, X_new):
        """
        Predicts new points. Mostly for visual exploration.
        """
        X_new = X_new.to(self.device)
        # Define likelihood
        with torch.no_grad():
            if self.standardize:
                X_new = self.Xscaler.transform(X_new)

            f_new = kernel(X_new, self.X1) @ self.alpha

            if self.standardize:
                f_new = self.Yscaler.inverse_transform(f_new)

        return f_new


if __name__ == "__main__":
    sigma2 = (torch.tensor(0.02)) ** 2

    gaussian_cost = GaussianCost(obs_var=sigma2)
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    device = "cpu"

    print(f"Using device: {device}")

    hyper_opt = KernelsHyperOpt(
        cost=gaussian_cost,
        kernel=kernel,
        device=device,
        standardize=False,
        reg_para=1e-7,
    )

    hyper_opt.fit(
        X=train_x,
        Y=train_y,
        batch_size=20,
        max_iter=100,
        lr1=1e-3,
        lr2=1e-3,
        n_hyper=0.2,
        patience=5,
        min_delta=0.0001,
    )

    f_new = hyper_opt.predict(train_x_all)

    print("lengthscale", kernel.base_kernel.lengthscale)
    # Create a plot
    plt.figure(figsize=(10, 6))

    # Plot training data
    plt.plot(train_x.numpy(), train_y.numpy(), "k.", label="Training Data")

    # Plot predictions
    plt.plot(train_x_all.numpy(), f_new.cpu().numpy(), "b-", label="Predicted Mean")

    # Add labels and title
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GP Predictions vs Training Data")
    plt.legend()

    # Show the plot
    plt.show()
