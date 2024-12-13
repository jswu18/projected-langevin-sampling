import math

import gpytorch
import matplotlib.pyplot as plt
import torch


# Define the Gaussian Process Model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()  # Zero mean for simplicity
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Define input points where you want to sample
train_x_all = torch.linspace(-20, 20, 150)
# Number of random entries to select
num_samples = 20

# Generate random indices
indices = torch.randperm(train_x_all.size(0))[:num_samples]
train_x = train_x_all[indices]

train_y = torch.zeros(train_x.size())  # We don't have training data
# Training data is 100 points in [0,1] inclusi
# ve regularly spaced
# train_x = torch.linspace(0, 1, 100)


def create_product_space(D, steps=100):
    # Create a list of D linspace tensors ranging from -1 to 1
    axes = [torch.linspace(-1, 1, steps) for _ in range(D)]

    # Create the meshgrid for all D dimensions
    grids = torch.meshgrid(*axes, indexing="ij")

    # Stack the grid to form a (steps^D, D) array
    product_space = torch.stack(grids, dim=-1)

    # Reshape the product space into (steps^D x D)
    reshaped_product_space = product_space.view(-1, D)

    return reshaped_product_space


# Example: Creating a 3D product space [-1,1] x [-1,1] x [-1,1]
# product_space_3d = create_product_space(D=3, steps=10)
# print(product_space_3d)  # Should output: torch.Size([1000000, 3])


# Define likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()

kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

# Instantiate the model
model = GPModel(train_x, train_y, likelihood, kernel)

# Set the model in evaluation mode (since we are sampling)
model.eval()

# Specify the RBF kernel lengthscale
model.covar_module.base_kernel.lengthscale = 1  # Set lengthscale

# Now, sample from the GP posterior
with torch.no_grad():
    train_y = model(train_x).sample()

if __name__ == "__main__":
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    # Print all parameters
    for param_name, param_value in kernel.named_parameters():
        print(f"{param_name}: {param_value.item()}")

    # Plot the sampled function
    plt.plot(train_x.numpy(), train_y.numpy())
    plt.title("Sample from Gaussian Process with RBF Kernel")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
