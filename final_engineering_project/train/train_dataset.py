from functools import lru_cache
import os
from typing import Any, Dict, List, Optional
import torch
import torchaudio  # type: ignore
import pandas as pd  # type: ignore
from torch.utils.data import Dataset
from final_engineering_project.train.OVectorUtility import OVectorUtility
from final_engineering_project.properties import train_path
from final_engineering_project.data.mixure_dataset import MixureDataset

_csv_file = os.path.join(train_path, "data.csv")
_root_dir = os.path.join(train_path, "files")

SampleType = Dict[str, Any]
EffectsInitType = Optional[List[List[str]]]


class TrainDataset(Dataset[SampleType]):
    def __init__(
        self,
        o_vector_utility: OVectorUtility,
        min_mixure: int,
        max_mixure: int,
        device: Any = None,
        from_fs: bool = True,
        length: int = 0,
    ) -> None:
        self._device = device
        self._o_vector_utility = o_vector_utility
        self._from_fs = from_fs
        self._length = length

        if from_fs:
            self._csv = pd.read_csv(_csv_file)
        else:
            self._mixure_dataset = MixureDataset(
                train_size=length,
                test_size=0,
                device=device,
                min_mixure=min_mixure,
                max_mixure=max_mixure,
            )

    def __len__(self) -> int:
        length = self._length

        if self._from_fs:
            csv_len = len(self._csv)

            if length == 0 or length > csv_len:
                return csv_len

        return length

    @lru_cache(maxsize=None)
    def __getitem__(self, idx: Any) -> SampleType:
        if torch.is_tensor(idx):  # type: ignore
            idx = idx.tolist()

        if self._from_fs:
            wav_path = os.path.join(_root_dir, self._csv.iloc[idx, 0])

            waveform, rate = torchaudio.sox_effects.apply_effects_file(
                path=wav_path,
                channels_first=True,
                effects=[],
            )

            events_wav_paths = [
                os.path.join(_root_dir, x) for x in self._csv.iloc[idx, 1].split("|")
            ]
            events_labels = self._csv.iloc[idx, 2].split("|")
            events_wavs = [
                torchaudio.sox_effects.apply_effects_file(
                    path=event_wav_path,
                    channels_first=True,
                    effects=[],
                )
                for event_wav_path in events_wav_paths
            ]

            sample = {
                "waveform": waveform.to(self._device) if self._device else waveform,
                "events": [
                    {
                        "waveform": event_wav[0].to(self._device)
                        if self._device
                        else event_wav[0],
                        "o_vector": self._o_vector_utility.get_o_vector_by_label(
                            events_labels[i]
                        ),
                    }
                    for i, event_wav in enumerate(events_wavs)
                ],
            }

            return sample

        mixure_sample = self._mixure_dataset[idx]
        waveform = mixure_sample["waveform"]
        events_wavs = mixure_sample["events"]
        events_labels = mixure_sample["labels"]

        sample = {
            "waveform": mixure_sample["waveform"],
            "events": [
                {
                    "waveform": event_wav,
                    "o_vector": self._o_vector_utility.get_o_vector_by_label(
                        events_labels[i]
                    ),
                }
                for i, event_wav in enumerate(events_wavs)
            ],
        }

        return sample
