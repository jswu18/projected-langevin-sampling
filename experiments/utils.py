import os
import shutil


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
