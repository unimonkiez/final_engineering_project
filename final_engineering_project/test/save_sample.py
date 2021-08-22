from final_engineering_project.train.OVectorUtility import OVectorUtility
import os
from os import path
from shutil import rmtree
import torch
import torchaudio
from torch.utils.data.dataloader import DataLoader
from final_engineering_project.train.model import Model
from final_engineering_project.properties import test_path

resample = 8000


def save_sample(
    dataloader: DataLoader,
    model: Model,
    o_vector_utility: OVectorUtility,
) -> None:
    cpu_device = torch.device("cpu")

    rmtree(path.join(test_path, "results"), ignore_errors=True)
    os.makedirs(path.join(test_path, "results", "files"), exist_ok=True)

    for (i, batch) in enumerate(dataloader):
        y = batch["waveform"].to(cpu_device)
        batch_size = y.shape[0]
        for b in range(batch_size):
            torchaudio.save(
                path.join(test_path, "results", "files", "{0}b{1}.wav".format(i, b)),
                y[b],
                resample,
            )

            for event_i, event in enumerate(batch["events"]):
                x = event["waveform"].to(cpu_device)
                o = event["o_vector"].to(cpu_device)
                label = o_vector_utility.get_label_by_o_vector(o[b])
                x_pred = model.to(cpu_device)(y, o)
                torchaudio.save(
                    path.join(
                        test_path,
                        "results",
                        "files",
                        "{0}b{1}_{2}_expected.wav".format(i, b, label),
                    ),
                    x[b],
                    resample,
                )
                torchaudio.save(
                    path.join(
                        test_path,
                        "results",
                        "files",
                        "{0}b{1}_{2}.wav".format(i, b, label),
                    ),
                    x_pred[b],
                    resample,
                )
