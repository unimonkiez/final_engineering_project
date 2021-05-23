import os
from os import path
from shutil import rmtree
import torch
import torchaudio
from torch.utils.data.dataloader import DataLoader
from final_engineering_project.test.TestDataset import TestDataset
from final_engineering_project.train.model import Model
from final_engineering_project.train.OVectorUtility import OVectorUtility
from final_engineering_project.properties import model_path, test_path

resample = 8000


def test() -> None:
    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    o_vector_utility = OVectorUtility(
        device=gpu_device,
    )

    model = Model(
        o_vector_length=o_vector_utility.get_vector_length(),
        device=gpu_device,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_dataset = TestDataset(
        o_vector_utility=o_vector_utility,
        device=gpu_device,
        from_fs=False,
        min_mixure=3,
        max_mixure=3,
        length=1,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
    )

    rmtree(path.join(test_path, "results"), ignore_errors=True)
    os.makedirs(path.join(test_path, "results", "files"), exist_ok=True)

    for (i, batch) in enumerate(test_dataloader):
        y = batch["waveform"]

        torchaudio.save(
            path.join(test_path, "results", "files", "{0}.wav".format(i)),
            y.squeeze(1).to(cpu_device),
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
                    "{0}_{1}_expected.wav".format(i, event_i),
                ),
                x.squeeze(1).to(cpu_device),
                resample,
            )
            torchaudio.save(
                path.join(
                    test_path, "results", "files", "{0}_{1}.wav".format(i, event_i)
                ),
                x_pred.squeeze(1).to(cpu_device),
                resample,
            )
