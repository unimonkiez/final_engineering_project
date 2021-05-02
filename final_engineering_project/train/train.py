import os
import torch
import torchaudio
from torchaudio.models.conv_tasnet import ConvTasNet
from torch.utils.data import DataLoader
from final_engineering_project.properties import model_path
from .solver import Solver
from .train_dataset import TrainDataset


def train() -> None:
    try:
        os.remove(model_path)
    except OSError:
        pass

    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cpu_device = torch.device("cpu")

    train_dataset = TrainDataset(
        device=gpu_device,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
    )

    model = ConvTasNet(
        num_sources=4,
        # encoder/decoder parameters
        enc_kernel_size=20,
        enc_num_feats=256,
        # mask generator parameters
        msk_kernel_size=3,
        msk_num_feats=256,
        msk_num_hidden_feats=512,
        msk_num_layers=8,
        msk_num_stacks=4,
    )

    solver = Solver(
        data=train_dataloader,
        model=model,
    )
    solver.train()
    torch.save(model, model_path)

    # for (batch_idx, batch) in enumerate(train_dataloader):
    #     print("\nBatch = " + str(batch_idx))
