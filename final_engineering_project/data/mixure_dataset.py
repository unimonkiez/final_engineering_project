from typing import Any, Dict, List
import torch
from numpy.lib import math
from torch.functional import Tensor
from torch.utils.data import Dataset, DataLoader
from .kaggle_dataset import KaggleDataset
from .random_dataset import RandomDataset
from .noise_dataset import NoiseDataset

SampleType = Dict[str, Any]

resample = 8000

resample_effect = ["rate", str(resample)]


def _get_label_to_kaggle_indecies(is_test=False) -> Dict[str, Tensor]:
    kaggle_dataset = KaggleDataset(no_load=True, is_test=is_test)
    label_to_kaggle_indices: Dict[str, Tensor] = {}
    dataloader = DataLoader(
        kaggle_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    for sample_index, sample in enumerate(dataloader):
        label = sample["label"][0]

        new_kaggle_indices = torch.tensor([sample_index])  # type: ignore
        if label in label_to_kaggle_indices:
            current_kaggle_indices = label_to_kaggle_indices[label]
            new_kaggle_indices = torch.cat(  # type: ignore
                (current_kaggle_indices, new_kaggle_indices),
                dim=0,
            )

        label_to_kaggle_indices[label] = new_kaggle_indices
    return label_to_kaggle_indices


def _normalize(signal: Tensor) -> Tensor:
    return signal / signal.std()


def _multiply_noise_by_SNR(signal: Tensor, noise: Tensor, desire_snr: float) -> Tensor:
    signal_ps = signal.var()
    noise_normalized = _normalize(noise)
    g = torch.sqrt(signal_ps / (10 ** (desire_snr / 10)))
    new_noise = g * noise_normalized
    # print(10*np.log10(signal_ps/np.var(new_noise)))
    return new_noise


def _get_events(
    gpu_device: Any,
    kaggle_dataset: KaggleDataset,
    # events_gain: float,
    kaggle_indecies: List[int],
    kaggle_starts_in_noise: List[float],
    kaggle_starts: List[float],
    kaggle_lengths: List[float],
) -> List[Dict[str, Any]]:
    events = []
    for i, kaggle_index in enumerate(kaggle_indecies):
        kaggle_start = kaggle_starts[i]
        kaggle_start_in_noise = kaggle_starts_in_noise[i]
        kaggle_length = kaggle_lengths[i]
        sample_length = kaggle_dataset.get_item_length_in_seconds(kaggle_index)

        kaggle_start_real = max(0, sample_length - kaggle_length) * kaggle_start
        kaggle_length_real = min(sample_length, kaggle_length)
        event = kaggle_dataset.get_item_with_effects(
            kaggle_index,
            [
                [
                    "trim",
                    format(kaggle_start_real, "f"),
                    format(kaggle_length_real, "f"),
                ],
            ],
        )
        waveform_gpu = _normalize(event["waveform"].to(gpu_device))
        waveform_length = len(waveform_gpu[0])
        zeros_to_pad_start = torch.zeros(
            1,
            math.ceil((6 - kaggle_length_real) * kaggle_start_in_noise * resample),
            device=gpu_device,
        )
        zeros_to_pad_end = torch.zeros(
            1,
            (resample * 6) - waveform_length - len(zeros_to_pad_start[0]),
            device=gpu_device,
        )
        waveform_gpu = torch.cat(
            (zeros_to_pad_start, waveform_gpu, zeros_to_pad_end),
            1,
        )
        event["waveform"] = waveform_gpu
        events.append(event)
    return events


class MixureDataset(Dataset[SampleType]):
    def __init__(
        self,
        train_size: int,
        test_size: int,
        min_mixure: int = 3,
        max_mixure: int = 3,
        device: Any = None,
    ) -> None:
        self._train_size = train_size
        self._test_size = test_size
        self._device = device

        self._noise_dataset = NoiseDataset(
            effects=[
                resample_effect,
            ],
        )
        self._kaggle_dataset_test = KaggleDataset(
            is_test=True,
            effects=[
                resample_effect,
            ],
        )
        self._kaggle_dataset_train = KaggleDataset(
            is_test=False,
            effects=[
                resample_effect,
            ],
        )
        label_to_kaggle_indices_test = _get_label_to_kaggle_indecies(is_test=True)
        label_to_kaggle_indices_train = _get_label_to_kaggle_indecies(is_test=False)
        kaggle_indices_test = list(label_to_kaggle_indices_test.values())
        kaggle_indices_train = list(label_to_kaggle_indices_train.values())

        self._random_dataset = RandomDataset(
            train_size=train_size,
            test_size=test_size,
            noise_size=len(self._noise_dataset),
            kaggle_indices_test=kaggle_indices_test,
            kaggle_indices_train=kaggle_indices_train,
            min_mixure=min_mixure,
            max_mixure=max_mixure,
        )

    def __len__(self) -> int:
        return self._train_size + self._test_size

    def __getitem__(self, idx: int) -> SampleType:
        sample_batched = self._random_dataset[idx]

        is_train = sample_batched["is_train"]
        noise_index = sample_batched["noise_index"].item()
        noise_start = sample_batched["noise_start"].item()
        events_gain = sample_batched["events_gain"].item()
        kaggle_indecies = sample_batched["kaggle_index"]
        kaggle_starts = sample_batched["kaggle_start"]
        kaggle_starts_in_noise = sample_batched["kaggle_start_in_noise"]
        kaggle_lengths = sample_batched["kaggle_length"]

        kaggle_dataset = (
            self._kaggle_dataset_train if is_train else self._kaggle_dataset_test
        )

        noise = self._noise_dataset.get_item_with_effects(
            noise_index,
            [
                # ["gain", "-n"],  # normalises to 0dB
                ["trim", str(noise_start), str(6)],
            ],
        )
        events = _get_events(
            gpu_device=self._device,
            kaggle_dataset=kaggle_dataset,
            kaggle_indecies=kaggle_indecies,
            kaggle_starts_in_noise=kaggle_starts_in_noise,
            kaggle_starts=kaggle_starts,
            kaggle_lengths=kaggle_lengths,
        )
        labels = list(set(map(lambda x: x["label"], events)))
        events_seperated_by_label = [
            [y for y in events if y["label"] == x] for x in labels
        ]
        class_events = [
            {
                "label": labels[i],
                "waveform": _normalize(
                    torch.stack(
                        [event["waveform"] for event in x],
                        dim=1,
                    ).sum(dim=1)
                ),
            }
            for i, x in enumerate(events_seperated_by_label)
        ]

        noise_waveform = noise["waveform"]
        noise_waveform_gpu = noise_waveform.to(self._device)
        single_channel_noise_waveform_gpu = torch.sum(
            noise_waveform_gpu,
            dim=0,
        ).reshape(1, resample * 6)
        events_waveform_gpu = _normalize(
            torch.stack(
                [event["waveform"] for event in class_events],
                dim=1,
            ).sum(dim=1)
        )
        snr_noise_waveform_gpu = _multiply_noise_by_SNR(
            signal=events_waveform_gpu,
            noise=single_channel_noise_waveform_gpu,
            desire_snr=events_gain,
        )
        waveform_gpu = torch.stack(
            [snr_noise_waveform_gpu, events_waveform_gpu],
            dim=1,
        ).sum(dim=1)

        sample = {
            "is_train": is_train,
            "waveform": waveform_gpu,
            "events": [event["waveform"] for event in class_events],
            "labels": labels,
        }

        return sample
