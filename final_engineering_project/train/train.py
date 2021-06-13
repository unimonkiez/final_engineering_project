from typing import Optional
from final_engineering_project.train.model import Model
from final_engineering_project.train.OVectorUtility import OVectorUtility
import os
import torch
from torch.utils.data import DataLoader
from final_engineering_project.properties import model_path, optimizer_path
from .solver import Solver
from .train_dataset import TrainDataset


def train(
    use_fs: bool,
    override_model: bool,
    size: int,
    batch_size: int,
    min_mixure: int,
    max_mixure: int,
    save_model_every: Optional[int],
    print_progress_every: Optional[int],
) -> None:
    if override_model:
        try:
            os.remove(model_path)
        except OSError:
            pass

    gpu_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # cpu_device = torch.device("cpu")

    o_vector_utility = OVectorUtility(
        device=gpu_device,
    )

    train_dataset = TrainDataset(
        o_vector_utility=o_vector_utility,
        device=gpu_device,
        from_fs=use_fs,
        min_mixure=min_mixure,
        max_mixure=max_mixure,
        length=size,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = Model(
        o_vector_length=o_vector_utility.get_vector_length(),
        device=gpu_device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4,
    )

    if not override_model:
        optimizer.load_state_dict(torch.load(optimizer_path))
        model.load_state_dict(torch.load(model_path))
        model.train()

    solver = Solver(
        data=train_dataloader,
        model=model,
        optimizer=optimizer,
        save_model_every=save_model_every,
        print_progress_every=print_progress_every,
    )
    solver.train()

    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(model.state_dict(), model_path)
