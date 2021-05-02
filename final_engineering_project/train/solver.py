import torch
from torch.utils.data import DataLoader


class Solver(object):
    def __init__(self, data: DataLoader, model: torch.nn.Module) -> None:
        self._data = data
        self._model = model

    def train(self) -> None:
        print("Training..")
