import os
from typing import Any, Dict, List, Optional
import torch
import torchaudio  # type: ignore
import pandas as pd  # type: ignore
from torch.utils.data import Dataset
from final_engineering_project.properties import noise_path

_csv_file = os.path.join(noise_path, "noise.csv")
_root_dir = os.path.join(noise_path, "NOISE")

SampleType = Dict[str, Any]
EffectsInitType = Optional[List[List[str]]]


class NoiseDataset(Dataset[SampleType]):
    def __init__(
        self,
        effects: EffectsInitType = None,
    ) -> None:
        self._csv = pd.read_csv(_csv_file)
        self._effects = effects if effects else []

    def __len__(self) -> int:
        return len(self._csv)

    def __getitem__(self, idx: Any) -> SampleType:
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()

        wav_path = os.path.join(_root_dir, self._csv.iloc[idx, 0])
        waveform, rate = torchaudio.sox_effects.apply_effects_file(
            path=wav_path,
            effects=self._effects,
            channels_first=True,
        )

        sample = {
            "waveform": waveform,
            "rate": rate,
        }

        return sample
