import os
from typing import Any
import pandas as pd  # type: ignore
from torch.functional import Tensor
import torch
from final_engineering_project.properties import kaggle_path

_csv_train_file = os.path.join(kaggle_path, "train.csv")


class OVectorUtility(object):
    def __init__(self, device: Any = None) -> None:
        self._device = device
        self._csv = pd.read_csv(_csv_train_file)
        self._label_list = list(set(self._csv.label))
        self._label_dict = {k: v for v, k in enumerate(self._label_list)}

    def get_vector_length(self) -> int:
        return len(self._label_list)

    def get_o_vector_by_label(self, label: str) -> Tensor:
        o_vector = torch.zeros(self.get_vector_length(), device=self._device)
        label_index = self._label_dict[label]
        o_vector[label_index] = 1

        return o_vector

    def get_label_by_o_vector(self, o_vector: Tensor) -> str:
        label_index = (o_vector == 1).nonzero(as_tuple=True)[0]
        label = self._label_list[label_index]

        return label
