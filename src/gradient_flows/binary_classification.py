from typing import Optional

import gpytorch
import torch

from src.gradient_flows.base import GradientFlowBase
from src.kernels import GradientFlowKernel
from src.samplers import sample_multivariate_normal


class GradientFlowBinaryClassification(GradientFlowBase):
    """
    N is the number of training points
    M is the number of inducing points
    P is the number of particles
    D is the dimensionality of the data
    """

    def __init__(
        self,
        number_of_particles: int,
        x_induce: torch.Tensor,
        y_train: torch.Tensor,
        x_train: torch.Tensor,
        kernel: GradientFlowKernel,
        observation_noise: float,
        y_induce: Optional[torch.Tensor] = None,
        jitter: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(
            number_of_particles=number_of_particles,
            x_induce=x_induce,
            y_train=y_train,
            x_train=x_train,
            kernel=kernel,
            observation_noise=observation_noise,
            y_induce=y_induce,
            jitter=jitter,
            seed=seed,
        )

    @staticmethod
    def sigmoid(y):
        # 1/(1+exp(-y))
        return torch.div(1, 1 + torch.exp(-y))

    def update(
        self,
        learning_rate: torch.Tensor,
    ) -> torch.Tensor:
        """
        N is the number of training points
        M is the number of inducing points
        P is the number of particles
        D is the dimensionality of the data
        """

        # K(Z, X) is (M, N)
        # A is (N, M)
        # U is (M, P)

        # U_{i+1} = U_{i} - \eta k(Z, X) [
        # Y(\phi(A U_i)) + (1-Y) (1-\phi(A U_i))
        # ] - \eta M k(Z, Z)^{-1} U_i + \sqrt(2\eta k(Z, Z)) \xi

        # with A := k(X, Z) k(Z, Z)^{-1}
        # \phi(x) = (1+exp(-y))^{-1} sigmoid function
        # \xi ~ N(0, I_M), maybe  e ~ N(0, k(Z, Z)) of size (M, P)? or sample from bernoulli instead of normal?

        # (N, P)
        prediction = self.sigmoid(
            gpytorch.solve(
                lhs=self.base_gram_induce_train.to_dense().T,
                input=self.base_gram_induce,
                rhs=self.particles,
            )
        )
        inverse_base_gram_particle_vector = gpytorch.solve(
            self.base_gram_induce, self.particles
        )  # k(Z, Z)^{-1} @ U(t) of size (M, P)
        noise_vector = sample_multivariate_normal(
            mean=torch.zeros(self.particles.shape[0]),
            cov=self.base_gram_induce,
            size=(self.number_of_particles,),
        ).T  # e ~ N(0, k(Z, Z)) of size (M, P)

        # - (eta/sigma^2) * K(Z, X) @ [
        #  torch.multiply(Y,  \phi(k(X, Z) @ k(Z, Z)^{-1} U(t))) (N, 1), (N, P)
        #   + torch.multiply((1-Y), (1 - \phi(k(X, Z) @ k(Z, Z)^{-1} U(t)))  (N, 1), (N, P)
        # ]
        # - eta * M * k(Z, Z)^{-1} @ U(t)
        # + sqrt(2 * eta) * e
        # size (M, P)

        particle_update = (
            -(learning_rate / self.observation_noise)
            * self.base_gram_induce_train
            @ (
                torch.mul(self.y_train[:, None], prediction)
                + torch.mul(1 - self.y_train[:, None], 1 - prediction)
            )
            - learning_rate
            * (self.x_induce.shape[0])
            * inverse_base_gram_particle_vector
            + torch.sqrt(2.0 * learning_rate) * noise_vector
        ).detach()
        self.particles += particle_update
        return particle_update.to_dense().detach()  # size (M, P)

    def predict_samples(
        self,
        x: torch.Tensor,
        include_observation_noise: bool = True,
    ) -> torch.Tensor:
        """
        Predicts samples for a given input.

        :param x: input of size (N*, D)
        :param include_observation_noise: whether to include observation noise in the sample
        :return: samples of size (N*, P)
        """
        gram_x_induce = self.kernel(
            x1=x,
            x2=self.x_induce,
        ).to_dense()  # r(x, Z) of size (N*, M)
        # gram_x = self.kernel(
        #     x1=x,
        #     x2=x,
        # ).to_dense()  # r(x, x) of size (N*, N*)
        # noise_vector = self._sample_predict_noise(
        #     gram_x=gram_x,
        #     gram_x_induce=gram_x_induce,
        #     number_of_samples=self.number_of_particles,
        #     include_observation_noise=include_observation_noise,
        # )  # e(x) of size (N*, P)
        #
        # # r(x, Z) @ r(Z, Z)^{-1} @ U(t) + e(x)
        # # size (N*, P)
        # return (
        #     gpytorch.solve(
        #         lhs=gram_x_induce,
        #         input=self.gram_induce,
        #         rhs=self.particles,
        #     )
        #     + noise_vector
        # )

        # G(x) + r(x, Z) @ r(Z, Z)^{-1} @ (U(t)-G(Z))
        # G([Z, x]) ~ N(0, r([Z, x], [Z, x]))
        zx = torch.concatenate((self.x_induce, x), dim=0)  # (M+N*, D)
        k_zx_zx = self.kernel(
            x1=zx,
            x2=zx,
        )  # (M+N*, M+N*)
        g_zx = sample_multivariate_normal(
            mean=torch.zeros(k_zx_zx.shape[0]),
            cov=k_zx_zx,
            size=(self.number_of_particles,),
        ).T  # (M+N*, P)
        return self.sigmoid(
            g_zx[self.x_induce.shape[0] :, :]
            + (
                gpytorch.solve(
                    lhs=gram_x_induce,
                    input=self.gram_induce,
                    rhs=(self.particles - g_zx[: self.x_induce.shape[0], :]),
                )
            )
        )

    def predict(
        self,
        x: torch.Tensor,
        jitter: float = 1e-20,
    ) -> gpytorch.distributions.MultivariateNormal:
        """
        Predicts the mean and variance for a given input.
        :param x: input of size (N*, D)
        :param jitter: jitter to add to the diagonal of the covariance matrix if it is not positive definite
        :return: normal distribution of size (N*,)
        """
        samples = self.predict_samples(x=x)
        return gpytorch.distributions.MultivariateNormal(
            mean=samples.mean(dim=1),
            covariance_matrix=torch.diag(torch.clip(samples.var(axis=1), jitter, None)),
        )
