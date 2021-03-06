from shutil import rmtree
import time
from os import path
from typing import Optional
import torch
import torchaudio
import os
import csv
from .mixure_dataset import MixureDataset
from final_engineering_project.properties import train_path, test_path

resample = 8000


def create_data(
    train_size: int,
    test_size: int,
    min_mixure: int,
    max_mixure: int,
    print_progress_every: Optional[int],
) -> None:
    previous_time = time.time()

    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    mixure_dataset = MixureDataset(
        train_size=train_size,
        test_size=test_size,
        device=gpu_device,
        min_mixure=min_mixure,
        max_mixure=max_mixure,
    )

    rmtree(train_path, ignore_errors=True)
    rmtree(test_path, ignore_errors=True)
    os.makedirs(path.join(train_path, "files"), exist_ok=True)
    os.makedirs(path.join(test_path, "files"), exist_ok=True)

    with open(path.join(train_path, "data.csv"), "w", newline="") as train_file:
        with open(path.join(test_path, "data.csv"), "w", newline="") as test_file:
            train_writer = csv.writer(train_file)
            test_writer = csv.writer(test_file)
            train_writer.writerow(["fname", "events", "labels"])
            test_writer.writerow(["fname", "events", "labels"])

            for i in range(len(mixure_dataset)):
                sample_batched = mixure_dataset[i]
                is_train = sample_batched["is_train"]
                waveform = sample_batched["waveform"]
                events = sample_batched["events"]
                labels = sample_batched["labels"]

                processed_path = train_path if is_train else test_path
                writer = train_writer if is_train else test_writer

                fname = "{0}.wav".format(i)
                events_str = "|".join(
                    ["{0}_{1}.wav".format(i, label) for label in labels]
                )
                labels_str = "|".join(labels)

                torchaudio.save(
                    path.join(processed_path, "files", "{0}".format(fname)),
                    waveform.to(cpu_device),
                    resample,
                )
                for x_i, x in enumerate(events):
                    torchaudio.save(
                        path.join(
                            processed_path,
                            "files",
                            "{0}_{1}.wav".format(i, labels[x_i]),
                        ),
                        x.to(cpu_device),
                        resample,
                    )

                writer.writerow([fname, events_str, labels_str])

                if print_progress_every is not None:
                    iteration = i + 1
                    if iteration % print_progress_every == 0:
                        now = time.time()
                        print(
                            "proccessed {number} of files, this batch took {diff} seconds.".format(
                                number=iteration,
                                diff=now - previous_time,
                            ),
                        )
                        previous_time = now
