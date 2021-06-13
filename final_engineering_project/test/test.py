from final_engineering_project.test.metric import get_snr
import os
from os import path
from shutil import rmtree
import time
from typing import Optional
import torch
from torch.utils.data.dataloader import DataLoader
from final_engineering_project.test.TestDataset import TestDataset
from final_engineering_project.train.model import Model
from final_engineering_project.train.OVectorUtility import OVectorUtility
from final_engineering_project.properties import model_path, test_path

resample = 8000


def test(
    size: int,
    batch_size: int,
    print_progress_every: Optional[int],
) -> None:
    gpu_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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
        length=size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    rmtree(path.join(test_path, "results"), ignore_errors=True)
    os.makedirs(path.join(test_path, "results", "files"), exist_ok=True)

    previous_time = time.time()
    previous_snr = 0
    total_number = len(test_dataloader)

    for (i, batch) in enumerate(test_dataloader):
        y = batch["waveform"]

        for event_i, event in enumerate(batch["events"]):
            x = event["waveform"]
            # x = torch.zeros_like(x)  # Test
            o = event["o_vector"]
            x_pred = model(y, o)
            previous_snr = get_snr(x_pred, x)

        iteration = i + 1
        if print_progress_every is not None:
            if iteration % print_progress_every == 0:
                now = time.time()
                print(
                    "Tested {number}/{total_number} of batches, SNR is {snr}, this batch took {diff} seconds.".format(
                        number=iteration,
                        total_number=total_number,
                        diff=now - previous_time,
                        snr=previous_snr.mean(),
                    ),
                )
                previous_time = now
