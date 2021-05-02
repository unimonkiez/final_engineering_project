from shutil import rmtree
import time
from os import path
from typing import Any, Dict, List
from numpy.lib import math
import torch
import torchaudio
import os
import csv
from torch.functional import Tensor
from .noise_dataset import NoiseDataset
from .kaggle_dataset import KaggleDataset
from .random_dataset import RandomDataset
from torch.utils.data import DataLoader
from final_engineering_project.properties import train_path, test_path


resample = 8000
train_size = 100
test_size = 10
print_every = 5

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


def normalize(signal: Tensor) -> Tensor:
    return signal / signal.std()


def multiply_noise_by_SNR(signal: Tensor, noise: Tensor, desire_snr: float) -> Tensor:
    signal_ps = signal.var()
    noise_normalized = normalize(noise)
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
        waveform_gpu = normalize(event["waveform"].to(gpu_device))
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


def create_data() -> None:
    previous_time = time.time()

    noise_dataset = NoiseDataset(
        effects=[
            resample_effect,
        ],
    )
    kaggle_dataset_test = KaggleDataset(
        is_test=True,
        effects=[
            resample_effect,
        ],
    )
    kaggle_dataset_train = KaggleDataset(
        is_test=False,
        effects=[
            resample_effect,
        ],
    )
    label_to_kaggle_indices_test = _get_label_to_kaggle_indecies(is_test=True)
    label_to_kaggle_indices_train = _get_label_to_kaggle_indecies(is_test=False)
    kaggle_indices_test = list(label_to_kaggle_indices_test.values())
    kaggle_indices_train = list(label_to_kaggle_indices_train.values())

    random_dataset = RandomDataset(
        train_size=train_size,
        test_size=test_size,
        noise_size=len(noise_dataset),
        kaggle_indices_test=kaggle_indices_test,
        kaggle_indices_train=kaggle_indices_train,
        min_mixure=3,
        max_mixure=5,
    )
    random_dataloader = DataLoader(
        random_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    rmtree(train_path, ignore_errors=True)
    rmtree(test_path, ignore_errors=True)
    os.makedirs(path.join(train_path, "files"), exist_ok=True)
    os.makedirs(path.join(test_path, "files"), exist_ok=True)

    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    with open(path.join(train_path, "data.csv"), "w", newline="") as train_file:
        with open(path.join(test_path, "data.csv"), "w", newline="") as test_file:
            train_writer = csv.writer(train_file)
            test_writer = csv.writer(test_file)
            train_writer.writerow(["fname", "events", "labels"])
            test_writer.writerow(["fname", "events", "labels"])

            for i, sample_batched in enumerate(random_dataloader):
                is_train = sample_batched["is_train"].item()
                noise_index = sample_batched["noise_index"].item()
                noise_start = sample_batched["noise_start"].item()
                events_gain = sample_batched["events_gain"].item()
                kaggle_indecies = sample_batched["kaggle_index"].tolist()[0]
                kaggle_starts = sample_batched["kaggle_start"].tolist()[0]
                kaggle_starts_in_noise = sample_batched[
                    "kaggle_start_in_noise"
                ].tolist()[0]
                kaggle_lengths = sample_batched["kaggle_length"].tolist()[0]

                processed_path = train_path if is_train else test_path
                writer = train_writer if is_train else test_writer
                kaggle_dataset = (
                    kaggle_dataset_train if is_train else kaggle_dataset_test
                )

                noise = noise_dataset.get_item_with_effects(
                    noise_index,
                    [
                        # ["gain", "-n"],  # normalises to 0dB
                        ["trim", str(noise_start), str(6)],
                    ],
                )
                events = _get_events(
                    gpu_device=gpu_device,
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
                        "waveform": normalize(
                            torch.stack(
                                [event["waveform"] for event in x],
                                dim=1,
                            ).sum(dim=1)
                        ),
                    }
                    for i, x in enumerate(events_seperated_by_label)
                ]

                noise_waveform = noise["waveform"]
                noise_waveform_gpu = noise_waveform.to(gpu_device)
                single_channel_noise_waveform_gpu = torch.sum(
                    noise_waveform_gpu,
                    dim=0,
                ).reshape(1, resample * 6)
                events_waveform_gpu = normalize(
                    torch.stack(
                        [event["waveform"] for event in class_events],
                        dim=1,
                    ).sum(dim=1)
                )
                snr_noise_waveform_gpu = multiply_noise_by_SNR(
                    signal=events_waveform_gpu,
                    noise=single_channel_noise_waveform_gpu,
                    desire_snr=events_gain,
                )
                waveform_gpu = torch.stack(
                    [snr_noise_waveform_gpu, events_waveform_gpu],
                    dim=1,
                ).sum(dim=1)
                waveform = waveform_gpu.to(cpu_device)

                fname = "{0}.wav".format(i)
                events_str = "|".join(
                    ["{0}_{1}.wav".format(i, label) for label in labels]
                )
                labels_str = "|".join(labels)

                # torchaudio.save(
                #     path.join(processed_path, "files", "noise_{0}".format(fname)),
                #     noise_waveform,
                #     resample,
                # )
                # torchaudio.save(
                #     path.join(
                #         processed_path,
                #         "files",
                #         "single_noise_{0}".format(fname),
                #     ),
                #     single_channel_noise_waveform_gpu.to(cpu_device),
                #     resample,
                # )
                # torchaudio.save(
                #     path.join(processed_path, "files", "events_{0}".format(fname)),
                #     events_waveform_gpu.to(cpu_device),
                #     resample,
                # )
                torchaudio.save(
                    path.join(processed_path, "files", "{0}".format(fname)),
                    waveform,
                    resample,
                )
                for x in class_events:
                    torchaudio.save(
                        path.join(
                            processed_path, "files", "{0}_{1}.wav".format(i, x["label"])
                        ),
                        x["waveform"].to(cpu_device),
                        resample,
                    )

                writer.writerow([fname, events_str, labels_str])

                iteration = i + 1
                if iteration % print_every == 0:
                    now = time.time()
                    print(
                        "proccessed {number} of files, this batch took {diff} seconds.".format(
                            number=iteration,
                            diff=now - previous_time,
                        ),
                    )
                    previous_time = now

                # break
