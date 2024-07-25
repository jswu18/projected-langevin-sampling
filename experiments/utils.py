import argparse
import os
import shutil
from typing import Union


def create_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    :param directory:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def remove_directory(directory: str) -> None:
    """
    Remove directory if it exists.
    :param directory:
    :return:
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)


def str2bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1"):
        return True
    elif value.lower() in ("false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
