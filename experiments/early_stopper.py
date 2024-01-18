import numpy as np


class EarlyStopper:
    """
    Early stopping class
    Adapted from https://stackoverflow.com/a/73704579
    """

    def __init__(self, patience: float = 1e-4):
        self.patience = patience
        self.simulation_time = 0
        self.min_loss = float("inf")

    def should_stop(self, loss: float, step_size: float) -> bool:
        if not np.isfinite(loss):
            return True
        elif loss >= self.min_loss:
            self.simulation_time += step_size
            return self.simulation_time >= self.patience
        else:
            self.min_loss = loss
            self.simulation_time = 0
            return False
