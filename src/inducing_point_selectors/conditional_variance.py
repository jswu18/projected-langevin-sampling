from typing import Optional, Tuple

import gpytorch
import numpy as np
import torch

from src.inducing_point_selectors.base import InducingPointSelector


class ConditionalVarianceInducingPointSelector(InducingPointSelector):
    """
    Adapted from https://github.com/markvdw/RobustGP/blob/master/robustgp/init_methods/methods.py
    """

    def __init__(
        self,
        threshold: Optional[float] = 0.0,
    ):
        """

        Args:
            threshold: float or None, if not None, if tr(Kff-Qff)<threshold, stop choosing inducing points as the approx
                       has converged.
        """
        self.threshold = threshold

    def compute_induce_data(
        self,
        x: torch.Tensor,
        m: int,
        kernel: gpytorch.kernels.Kernel,
        jitter: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects inducing points based on the variance of the GP at the training points.

        Docstring from original code:
            The version of this code without sampling follows the Greedy approximation to MAP for DPPs in
            @incollection{NIPS2018_7805,
                    title = {Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity},
                    author = {Chen, Laming and Zhang, Guoxin and Zhou, Eric},
                    booktitle = {Advances in Neural Information Processing Systems 31},
                    year = {2018},
                }
            and the initial code is based on the implementation of this algorithm (https://github.com/laming-chen/fast-map-dpp)
            It is equivalent to running a partial pivoted Cholesky decomposition on Kff (see Figure 2 in the below ref.),
            @article{fine2001efficient,
                    title={Efficient SVM training using low-rank kernel representations},
                    author={Fine, Shai and Scheinberg, Katya},
                    journal={Journal of Machine Learning Research},
                    year={2001}
                }

            Initializes based on variance of noiseless GP fit on inducing points currently in active set
            Complexity: O(NM) memory, O(NM^2) time

        """
        assert m > 1, "Must have at least 2 inducing points"
        number_of_training_points = x.shape[0]
        perm = np.random.permutation(
            number_of_training_points
        )  # permute entries so tie-breaking is random
        x = x[perm, ...]
        # note this will throw an out of bounds exception if we do not update each entry
        indices = np.zeros(m, dtype=int) + number_of_training_points
        gram = kernel(
            x1=x,
            x2=x,
        )
        gram = gram if gram.ndim == 2 else gram[0, :, :]
        di = gram.cpu().diagonal().detach().numpy() + jitter
        indices[0] = np.argmax(di)  # select first point, add to index 0
        ci = np.zeros(
            (m - 1, number_of_training_points)
        )  # [m,number_of_training_points]
        for i in range(m - 1):
            j = int(indices[i])  # int
            new_induce_data = x[j : j + 1]  # [1,D]
            dj = np.sqrt(di[j])  # float
            cj = ci[:i, j]  # [i, 1]
            gram_matrix_raw = (
                kernel(
                    x1=x,
                    x2=new_induce_data,
                )
                .cpu()
                .detach()
                .numpy()
            )
            gram_matrix_raw = (
                gram_matrix_raw
                if gram_matrix_raw.ndim == 2
                else gram_matrix_raw[0, :, :]
            )
            gram_matrix = np.round(np.squeeze(gram_matrix_raw), 20)  # [N]
            gram_matrix[j] += jitter
            ei = (gram_matrix - np.dot(cj, ci[:i])) / dj
            ci[i, :] = ei
            try:
                di -= ei**2
            except FloatingPointError:
                pass
            di = np.clip(di, 0, None)
            # added to original code to prevent picking the same point twice
            indices = indices.astype(int)
            for next_idx in reversed(np.argsort(di)):
                if int(next_idx) not in indices[: i + 1]:
                    indices[i + 1] = int(next_idx)
                    break
            # sum of di is tr(Kff-Qff), if this is small things are ok
            if np.sum(np.clip(di, 0, None)) < self.threshold:
                indices = indices[:m]
                print(
                    "ConditionalVariance: Terminating selection of inducing points early."
                )
                break
        indices = indices.astype(int)
        induce_data = x[indices]
        indices = perm[indices]
        return induce_data, torch.from_numpy(indices)
