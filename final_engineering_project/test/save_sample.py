import os
from os import path
from shutil import rmtree
import torch
import torchaudio
from torch.utils.data.dataloader import DataLoader
from final_engineering_project.train.model import Model
from final_engineering_project.properties import test_path

resample = 8000


def save_sample(dataloader: DataLoader, model: Model) -> None:
    cpu_device = torch.device("cpu")

    rmtree(path.join(test_path, "results"), ignore_errors=True)
    os.makedirs(path.join(test_path, "results", "files"), exist_ok=True)

    for (i, batch) in enumerate(dataloader):
        y = batch["waveform"]
        batch_size = y.shape[0]
        for b in range(batch_size):
            torchaudio.save(
                path.join(test_path, "results", "files", "{0}b{1}.wav".format(i, b)),
                y[b].to(cpu_device),
                resample,
            )

            for event_i, event in enumerate(batch["events"]):
                x = event["waveform"]
                o = event["o_vector"]
                x_pred = model(y, o)
                torchaudio.save(
                    path.join(
                        test_path,
                        "results",
                        "files",
                        "{0}b{1}_{2}_expected.wav".format(i, b, event_i),
                    ),
                    x[b].to(cpu_device),
                    resample,
                )
                torchaudio.save(
                    path.join(
                        test_path,
                        "results",
                        "files",
                        "{0}b{1}_{2}.wav".format(i, b, event_i),
                    ),
                    x_pred[b].to(cpu_device),
                    resample,
                )
