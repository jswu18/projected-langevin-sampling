import os
import random

import numpy as np
import torch


def get_torch_generator(
    seed: int | None,
    device: torch.device | None = None,
) -> torch.Generator | None:
    if seed is None:
        return None
    if device is None or device.type == "cpu":
        return torch.Generator().manual_seed(seed)
    return torch.Generator(device=device).manual_seed(seed)


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for
     all packages used in the project.
    https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
