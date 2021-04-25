import os
from typing import Any, Dict, List, Optional
import torch
import torchaudio  # type: ignore
import pandas as pd  # type: ignore
from torch.utils.data import Dataset
from final_engineering_project.properties import kaggle_path

_csv_file = os.path.join(kaggle_path, "train.csv")
_root_dir = os.path.join(kaggle_path, "audio_train")

SampleType = Dict[str, Any]
EffectsInitType = Optional[List[List[str]]]


class KaggleDataset(Dataset[SampleType]):
    def __init__(
        self,
        no_load: bool = False,
        effects: EffectsInitType = None,
    ) -> None:
        self._csv = pd.read_csv(_csv_file)
        self._no_load = no_load
        self._effects = effects if effects else []

    def __len__(self) -> int:
        return len(self._csv)

    def get_item_length_in_seconds(self, idx: Any) -> float:
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()
        wav_path = os.path.join(_root_dir, self._csv.iloc[idx, 0])
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
            wav_path = os.path.join(_root_dir, self._csv.iloc[idx, 0])
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

    def __getitem__(self, idx: Any) -> SampleType:
        return self.get_item_with_effects(idx, [])
