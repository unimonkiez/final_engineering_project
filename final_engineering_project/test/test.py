import torch
from final_engineering_project.properties import model_path


def test() -> None:
    model = torch.load(model_path)
    print(model)
