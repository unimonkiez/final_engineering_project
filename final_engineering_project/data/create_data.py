from shutil import rmtree
import time
from os import path
from numpy.lib import math
import torch
import torchaudio
import os
import csv
from torch.functional import Tensor
from .mixure_dataset import MixureDataset
from .kaggle_dataset import KaggleDataset
from .random_dataset import RandomDataset
from torch.utils.data import DataLoader
from final_engineering_project.properties import train_path, test_path

resample = 8000

train_size = 100
test_size = 10
print_every = 5


def create_data() -> None:
    previous_time = time.time()

    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    mixure_dataset = MixureDataset(
        train_size=train_size,
        test_size=test_size,
        device=gpu_device,
    )
    mixure_dataloader = DataLoader(
        mixure_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
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

            for i, sample_batched in enumerate(mixure_dataloader):
                is_train = sample_batched["is_train"].item()
                waveform = sample_batched["waveform"]
                events = sample_batched["events"]
                labels = [x[0] for x in sample_batched["labels"]]

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
                for x in events:
                    torchaudio.save(
                        path.join(
                            processed_path, "files", "{0}_{1}.wav".format(i, x["label"])
                        ),
                        x.to(cpu_device),
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
