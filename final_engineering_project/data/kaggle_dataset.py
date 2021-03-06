from functools import lru_cache
import os
from typing import Any, Dict, List, Optional
import torch
import torchaudio  # type: ignore
import pandas as pd  # type: ignore
from torch.utils.data import Dataset
from final_engineering_project.properties import kaggle_path

_csv_train_file = os.path.join(kaggle_path, "train.csv")
_csv_test_file = os.path.join(kaggle_path, "test.csv")
_train_root_dir = os.path.join(kaggle_path, "audio_train")
_test_root_dir = os.path.join(kaggle_path, "audio_test")

SampleType = Dict[str, Any]
EffectsInitType = Optional[List[List[str]]]


class KaggleDataset(Dataset[SampleType]):
    def __init__(
        self,
        is_test: bool = False,
        no_load: bool = False,
        effects: EffectsInitType = None,
    ) -> None:
        self._csv = pd.read_csv(_csv_test_file if is_test else _csv_train_file)
        self._root_dir = _test_root_dir if is_test else _train_root_dir
        self._no_load = no_load
        self._effects = effects if effects else []
        # if not no_load:
        # self._load_all_to_cache()

    def _load_all_to_cache(self) -> None:
        length = self.__len__()
        for idx in range(length):
            self.get_item_length_in_seconds(idx)
            self.__getitem__(idx)

    def __len__(self) -> int:
        return len(self._csv)

    @lru_cache(maxsize=None)
    def get_item_length_in_seconds(self, idx: Any) -> float:
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()
        wav_path = os.path.join(self._root_dir, self._csv.iloc[idx, 0])
        info = torchaudio.info(wav_path)

        return info.num_frames / info.sample_rate

    def get_item_with_effects(self, idx: Any, effects: List[List[str]]) -> SampleType:
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()

        label = self._csv.iloc[idx, 1]

        sample = None
        if self._no_load:
            sample = {
                "label": label,
            }
        else:
            wav_path = os.path.join(self._root_dir, self._csv.iloc[idx, 0])
            waveform, rate = torchaudio.sox_effects.apply_effects_file(
                path=wav_path,
                effects=self._effects + effects,
                channels_first=True,
            )

            sample = {
                "waveform": waveform,
                "rate": rate,
                "label": label,
            }

        return sample

    @lru_cache(maxsize=None)
    def __getitem__(self, idx: Any) -> SampleType:
        return self.get_item_with_effects(idx, [])
