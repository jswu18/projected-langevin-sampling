from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Data:
    x: torch.Tensor
    y: Optional[torch.Tensor] = None
    y_untransformed: Optional[torch.Tensor] = None
    name: str = "data"


@dataclass
class ExperimentData:
    name: str
    full: Data
    train: Optional[Data] = None
    test: Optional[Data] = None
    validation: Optional[Data] = None
    y_mean: torch.float = 0.0
    y_std: torch.float = 1.0

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str):
        experiment_data = torch.load(path)
        experiment_data.full.x
        experiment_data.full.y
        if experiment_data.full.y_untransformed is not None:
            experiment_data.full.y_untransformed
        experiment_data.full.name = "full"
        if experiment_data.train is not None:
            experiment_data.train.x
            experiment_data.train.y
            if experiment_data.train.y_untransformed is not None:
                experiment_data.train.y_untransformed
            experiment_data.train.name = "train"
        if experiment_data.test is not None:
            experiment_data.test.x
            experiment_data.test.y
            if experiment_data.test.y_untransformed is not None:
                experiment_data.test.y_untransformed
            experiment_data.test.name = "test"
        if experiment_data.validation is not None:
            experiment_data.validation.x
            experiment_data.validation.y
            if experiment_data.validation.y_untransformed is not None:
                experiment_data.validation.y_untransformed
            experiment_data.validation.name = "validation"
        experiment_data.y_mean = float(experiment_data.y_mean.detach().item())
        experiment_data.y_std = float(experiment_data.y_std.detach().item())
        return experiment_data
