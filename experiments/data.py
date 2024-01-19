import enum
from dataclasses import dataclass
from typing import Optional

import torch


class ProblemType(str, enum.Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass
class Data:
    x: torch.Tensor
    y: Optional[torch.Tensor] = None
    y_untransformed: Optional[torch.Tensor] = None
    name: str = "data"


@dataclass
class ExperimentData:
    name: str
    problem_type: ProblemType
    full: Data
    train: Optional[Data] = None
    test: Optional[Data] = None
    validation: Optional[Data] = None
    y_mean: torch.float = 0.0
    y_std: torch.float = 1.0

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str, problem_type: ProblemType):
        experiment_data = torch.load(path)
        experiment_data.full.name = "full"
        experiment_data.problem_type = problem_type
        if experiment_data.train is not None:
            experiment_data.train.name = "train"
        if experiment_data.test is not None:
            experiment_data.test.name = "test"
        if experiment_data.validation is not None:
            experiment_data.validation.name = "validation"
        if isinstance(experiment_data.y_mean, torch.Tensor):
            experiment_data.y_mean = float(experiment_data.y_mean.detach().item())
        if isinstance(experiment_data.y_std, torch.Tensor):
            experiment_data.y_std = float(experiment_data.y_std.detach().item())
        return experiment_data
