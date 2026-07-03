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

    @staticmethod
    def _move_tensor(
        tensor: torch.Tensor,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        target_dtype = (
            dtype if dtype is not None and tensor.is_floating_point() else None
        )
        return tensor.to(device=device, dtype=target_dtype)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Data":
        self.x = self._move_tensor(self.x, device=device, dtype=dtype)
        if self.y is not None:
            self.y = self._move_tensor(self.y, device=device, dtype=dtype)
        if self.y_untransformed is not None:
            self.y_untransformed = self._move_tensor(
                self.y_untransformed,
                device=device,
                dtype=dtype,
            )
        return self


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

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ExperimentData":
        self.full.to(device=device, dtype=dtype)
        if self.train is not None:
            self.train.to(device=device, dtype=dtype)
        if self.test is not None:
            self.test.to(device=device, dtype=dtype)
        if self.validation is not None:
            self.validation.to(device=device, dtype=dtype)
        if isinstance(self.y_mean, torch.Tensor):
            self.y_mean = self.y_mean.to(device=device, dtype=dtype)
        if isinstance(self.y_std, torch.Tensor):
            self.y_std = self.y_std.to(device=device, dtype=dtype)
        return self

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str, problem_type: ProblemType):
        experiment_data = torch.load(path, weights_only=False)
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
