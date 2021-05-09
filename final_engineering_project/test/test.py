import torch
from final_engineering_project.train.model import Model
from final_engineering_project.train.OVectorUtility import OVectorUtility
from final_engineering_project.properties import model_path


def test() -> None:
    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # cpu_device = torch.device("cpu")

    o_vector_utility = OVectorUtility(
        device=gpu_device,
    )

    model = Model(
        o_vector_length=o_vector_utility.get_vector_length(),
        device=gpu_device,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print(model)
