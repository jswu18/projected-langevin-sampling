import os


def create_directory(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    :param directory:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
