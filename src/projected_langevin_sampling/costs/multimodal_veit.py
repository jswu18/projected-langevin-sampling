from typing import Tuple

import gpytorch
import torch
import sys
import os 
import matplotlib.pyplot as plt

#Uncomment this James. I dont know why I always have to add bs like that for relative imports to work
sys.path.append(os.path.abspath('/Users/vdw/Library/CloudStorage/OneDrive-Personal/Code/function-space-gradient-flow'))

from src.projected_langevin_sampling.costs.base import PLSCost
from src.projected_langevin_sampling.link_functions import (
    IdentityLinkFunction,
    PLSLinkFunction,
)


class MultiModalCostVeit(PLSCost):
    """
    N is the number of training points.
    M is the dimensionality of the function space approximation.
    J is the number of particles.
    D is the dimensionality of the data.
    """

    def __init__(
        self,
        observation_noise: float,
        shift: float,
        bernoulli_noise: float,
        y_train: torch.Tensor,
        link_function: PLSLinkFunction,
    ):
        super().__init__(
            link_function=link_function, observation_noise=observation_noise #observation noise is standard deviation
        )
        self.shift = shift
        self.bernoulli_noise = bernoulli_noise
        self.y_train = y_train


    def predict(
        self,
        prediction_samples: torch.Tensor,
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Constructs a multivariate normal distribution from the prediction samples.
        :param prediction_samples: The prediction samples of size (N, J).
        :return: The multivariate normal distribution.
        """
        pass

    def calculate_cost(
        self, 
        untransformed_train_prediction_samples: torch.Tensor,
        reduction = True
    ) -> torch.Tensor:
        """
        Calculates the negative log likelihood cost for the current particles. This method takes the untransformed train prediction
        samples calculated with the current particles. This is implemented in the basis class of PLS.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: 
        - The cost of size (J,) for each particle if red=True
        - The cost of size (N,J) for each particle at each observation
        """
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )

        # (N, J)
        residuals = self.y_train[:, None] - train_prediction_samples

        errors_mode_1 = residuals - self.shift # [N,J]
        # y - f(x)
        errors_mode_2 = residuals # [N,J]

        # (N, J)
        log_likelihood_mode_1 = -0.5 * (
            torch.square(errors_mode_1) / (self.observation_noise**2)
        ) - torch.log(
            torch.sqrt(2 * torch.tensor([torch.pi]) * (self.observation_noise**2))
        )

        log_likelihood_mode_2 = -0.5 * (
            torch.square(errors_mode_2) / (self.observation_noise**2)
        ) - torch.log(
            torch.sqrt(2 * torch.tensor([torch.pi]) * (self.observation_noise**2))
        )

        nll_multimodal = -torch.logsumexp(
            torch.stack(
                [
                    torch.log(torch.tensor(self.bernoulli_noise))
                    + log_likelihood_mode_1,
                    torch.log(torch.tensor(1 - self.bernoulli_noise))
                    + log_likelihood_mode_2,
                ]
            ),
            dim=0,
        )

        if reduction:
            return nll_multimodal.sum(axis=0)

        else:
            return nll_multimodal
        
    def calculate_cost_derivative(
         self, untransformed_train_prediction_samples: torch.Tensor
     ) -> torch.Tensor:
        """
        This method is used when the link function is the identity.
        :param untransformed_train_prediction_samples: The untransformed train prediction samples of size (N, J).
        :return: The cost derivative of size (N, J).
        """
        train_prediction_samples = self.link_function(
            untransformed_train_prediction_samples
        )

        #[N.J]
        cost = self.calculate_cost(train_prediction_samples,reduction=False)
        residual = self.y_train[:, None]-train_prediction_samples

        # (N, J)
        errors_mode_1 = residual - self.shift
        errors_mode_2 = residual

        likelihood_mode_1_unscaled = torch.exp(
            -0.5
            * torch.square(errors_mode_1)
            / (self.observation_noise**2)
            + cost
        )

        likelihood_mode_2_unscaled = torch.exp(
            -0.5
            * torch.square(errors_mode_2)
            / (self.observation_noise**2)
            + cost
        )

        res = self.bernoulli_noise * errors_mode_1 * likelihood_mode_1_unscaled+ (1 - self.bernoulli_noise) * errors_mode_2 * likelihood_mode_2_unscaled
        res /= torch.sqrt(2 * torch.tensor([torch.pi])) * (self.observation_noise**3)


        
        return -res




        # (N, J)
        #return -torch.divide(
        #    self.bernoulli_noise * errors_mode_1 * likelihood_mode_1_unscaled
        #    + (1 - self.bernoulli_noise) * errors_mode_2 * likelihood_mode_2_unscaled,
        #    torch.sqrt(2 * torch.tensor([torch.pi]))
        #    * (self.observation_noise**3)
        #    * likelihood,
        #)        




# Plotting function
def plot_costs(x,matrix):
    """
    Plots x, matrix[:,j] for j=1,...,J entries horizontally on top of each other with respect to the J dimension.
    
    Args:
    matrix (torch.Tensor): A NxJ tensor where N is the number of rows and J is the number of columns.
    """
    N, J = matrix.shape
    fig, ax = plt.subplots()

    # Loop over each row in the matrix and plot it horizontally with some offset in the y-axis
    for j in range(J):
        ax.plot(x, matrix[:, j], label=f'Cost of function {j+1}' if N <= 10 else '')

    ax.set_xlabel('Residual values')
    ax.set_ylabel('Cost values')
    ax.set_title(f'Plot of {J} functions with {N} observations each')
    plt.show()


if __name__ == '__main__':
    sigma = 0.01
    shift = 10
    N_train=1000
    J=1
    mixture_weight = 0.5
    y_train = torch.linspace(-10,20,steps=N_train)
    print(y_train)
    #y_train =torch.normal(mean=10, std=0.1, size=(N_train,1))

    func_values = torch.zeros(size=(N_train,1))
    #print(func_values)

    multimodal_nll = MultiModalCostVeit(
        observation_noise=sigma,
        shift=shift,
        bernoulli_noise=mixture_weight,
        y_train=y_train,
        link_function=IdentityLinkFunction()

    )

    cost = multimodal_nll.calculate_cost(func_values,reduction=False) #[N,J]   
    costs_derivative = multimodal_nll.calculate_cost_derivative(func_values) #[N,J]
    
    combined_tensor = torch.cat((cost, costs_derivative), dim=1)
    
    plot_costs(y_train,combined_tensor)