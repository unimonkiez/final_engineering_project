from typing import Dict
import torch
from torch.functional import Tensor
from .noise_dataset import NoiseDataset
from .kaggle_dataset import KaggleDataset
from .random_dataset import RandomDataset
from torch.utils.data import DataLoader

resample = 8000
train_size = 50000
test_size = 10000

resample_effect = ["rate", str(resample)]


def _get_label_to_kaggle_indecies() -> Dict[str, Tensor]:
    kaggle_dataset = KaggleDataset(no_load=True)
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


def create_data() -> None:
    noise_dataset = NoiseDataset(
        effects=[
            resample_effect,
        ],
    )
    kaggle_dataset = KaggleDataset(
        effects=[
            resample_effect,
        ],
    )
    label_to_kaggle_indices = _get_label_to_kaggle_indecies()
    kaggle_indices = list(label_to_kaggle_indices.values())
    random_dataset = RandomDataset(
        train_size=train_size,
        test_size=test_size,
        noise_size=len(noise_dataset),
        kaggle_indices=kaggle_indices,
        min_mixure=3,
        max_mixure=3,
    )
    random_dataloader = DataLoader(
        random_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    for _, sample_batched in enumerate(random_dataloader):
        print(sample_batched)
        break
