import torch
from abc import ABC, abstractmethod

class Costs(ABC):

    @abstractmethod
    def __call__(self, Y: torch.Tensor, Yhat: torch.Tensor) -> torch.Tensor:
        """
        Compute the cost (loss function).
        :param Y: Actual values (target)
        :param Yhat: Predicted values
        :return: Scalar cost (0-D tensor)
        """
        pass

    @abstractmethod
    def derivative(self, Y: torch.Tensor, Yhat: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the cost function with respect to Yhat.
        :param Y: Actual values (target)
        :param Yhat: Predicted values
        :return: Gradient of cost (same shape as Yhat)
        """
        pass

class GaussianCost(Costs):

    def __init__(self,obs_var:torch.Tensor) -> None:
        self.obs_var = obs_var

    
    def __call__(self, Y:torch.Tensor,Yhat:torch.Tensor):
        '''
        Calculates the negative log-likelihood for p(y|x) = N( y | f(x) , sigma2) with yhat = f(x)
        -  Y: (N,)
        - Yhat: (N,)
        - return: scalar
        '''
        N = Y.shape[0]

        term1 = N/2 * torch.log(2*torch.pi * self.obs_var)
        term2 = 1/(2*self.obs_var) * torch.sum((Y-Yhat)**2)

        return term1 + term2
    
    def derivative(self, Y:torch.Tensor,Yhat:torch.Tensor):
        '''
        Calculates the derrivative of the negative log-likelihood for p(y|x) = N( y | f(x) , sigma2) with yhat = f(x)
        -  Y: (N,)
        - Yhat: (N,)
        - return: (N,)
        '''        

        return -1/(self.obs_var) * (Y-Yhat)


