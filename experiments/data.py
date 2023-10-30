from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Data:
    x: torch.Tensor
    y: Optional[torch.Tensor] = None
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
