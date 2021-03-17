import torch


def start() -> None:
    print("PyTorch version is: {0}".format(torch.__version__))  # type: ignore
