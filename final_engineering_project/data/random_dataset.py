from typing import Any, Dict, List
import torch
from torch.functional import Tensor
from torch.utils.data import Dataset

SampleType = Dict[str, Any]


kaggle_min_length_seconds = 1.5
kaggle_max_length_seconds = 3
number_of_kaggle_for_mix = 6

kaggle_diff_length = kaggle_max_length_seconds - kaggle_min_length_seconds


def get_sample_sizes(total_mix: int, number_of_groups: int) -> Tensor:
    sample_sizes = torch.zeros(number_of_groups)
    for i in range(total_mix):
        index_of_group = i % number_of_groups
        sample_sizes[index_of_group] += 1
    return sample_sizes


class RandomDataset(Dataset[SampleType]):
    def __init__(
        self,
        train_size: int,
        test_size: int,
        noise_size: int,
        min_mixure: int,
        max_mixure: int,
        kaggle_indices: List[Tensor],
    ) -> None:
        self._train_size = train_size
        self._test_size = test_size
        self._noise_size = noise_size
        self._min_mixure = min_mixure
        self._max_mixure = max_mixure
        self._kaggle_indices = kaggle_indices
        self._kaggle_size = torch.tensor([len(indices) for indices in kaggle_indices])

    def __len__(self) -> int:
        return self._train_size + self._test_size

    def __getitem__(self, idx: int) -> SampleType:
        mixures = torch.randint(
            low=self._min_mixure,
            high=self._max_mixure + 1,
            size=(1,),
        )
        sample_sizes = get_sample_sizes(number_of_kaggle_for_mix, mixures.item())
        mixure_indecies = torch.randint(high=len(self._kaggle_size), size=(mixures,))
        kaggle_index = torch.cat(
            [
                (
                    self._kaggle_indices[mixure_index][
                        torch.randint(
                            high=self._kaggle_size[mixure_index],
                            size=(int(sample_sizes[i].item()),),
                        )
                    ]
                )
                for i, mixure_index in enumerate(mixure_indecies)
            ]
        )

        is_train = idx < self._train_size
        noise_index = torch.randint(high=self._noise_size, size=(1,))  # type: ignore
        noise_start = torch.rand(1) * 24
        kaggle_start = torch.rand(6)
        kaggle_start_in_noise = torch.rand(6)
        kaggle_length = (torch.rand(6) * kaggle_diff_length) + kaggle_min_length_seconds
        events_gain = (torch.rand(1) * 10) + 15

        sample = {
            "is_train": is_train,
            "noise_index": noise_index,
            "kaggle_index": kaggle_index,
            "noise_start": noise_start,
            "kaggle_start": kaggle_start,
            "kaggle_start_in_noise": kaggle_start_in_noise,
            "kaggle_length": kaggle_length,
            "events_gain": events_gain,
        }

        return sample
