import enum
from dataclasses import dataclass

import torch


class ProblemType(str, enum.Enum):
    POISSON_REGRESSION = "poisson_regression"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    MULTIMODAL_REGRESSION = "multimodal_regression"


@dataclass
class Data:
    x: torch.Tensor
    y: torch.Tensor | None = None
    y_untransformed: torch.Tensor | None = None
    name: str = "data"

    def __post_init__(self):
        if torch.cuda.is_available():
            self.x = self.x.to(device="cuda")
            if self.y is not None:
                self.y = self.y.to(device="cuda")
            if self.y_untransformed is not None:
                self.y_untransformed = self.y_untransformed.to(device="cuda")


@dataclass
class ExperimentData:
    name: str
    problem_type: ProblemType
    full: Data
    train: Data | None = None
    test: Data | None = None
    validation: Data | None = None
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
