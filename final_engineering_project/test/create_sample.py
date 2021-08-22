from final_engineering_project.test.save_sample import save_sample
import torch
from torch.utils.data.dataloader import DataLoader
from final_engineering_project.test.TestDataset import TestDataset
from final_engineering_project.train.model import Model
from final_engineering_project.train.OVectorUtility import OVectorUtility
from final_engineering_project.properties import model_path

resample = 8000


def create_sample() -> None:
    gpu_device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    o_vector_utility = OVectorUtility(
        device=gpu_device,
    )

    model = Model(
        o_vector_length=o_vector_utility.get_vector_length(),
    ).to(gpu_device)
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
    save_sample(
        dataloader=test_dataloader,
        model=model,
        o_vector_utility=o_vector_utility,
    )
