from dataclasses import dataclass

import gpytorch
import numpy as np
import torch


@dataclass
class NonParametric:
    samples: torch.Tensor  # Having shape (number of points, number of samples)

    @property
    def mean(self):
        return self.samples.mean(axis=1)

    def quantile(self, q: float):
        return self.samples.quantile(q, dim=1)


@dataclass
class StudentTMarginals:
    """
    A class for the marginals of the Student T distribution where
    the degrees of freedom is the same for all marginals and the
    loc and scale are different for each individual Student T distribution.
    """

    df: float
    loc: torch.Tensor  # Having shape (n,)
    scale: torch.Tensor  # Having shape (n,)

    def negative_log_likelihood(self, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the average log probability of the y values
        given the Student T marginals.
        :param y: The y values of shape (n,).
        :return: The average log probability.
        """
        return -torch.tensor(
            np.mean(
                [
                    gpytorch.distributions.base_distributions.StudentT(
                        df=self.df,
                        loc=loc_,
                        scale=scale_,
                    )
                    .log_prob(y_)
                    .cpu()
                    .detach()
                    .numpy()
                    for loc_, scale_, y_ in zip(self.loc, self.scale, y)
                ]
            )
        )
