from final_engineering_project.data.main import create_data
from final_engineering_project.train.train import train
from final_engineering_project.test.test import test


def step_1() -> None:
    create_data()


def step_2() -> None:
    train()


def step_3() -> None:
    test()


def start() -> None:
    # print("PyTorch version is: {0}".format(torch.__version__))  # type: ignore
    step_1()


# async def download_async() -> None:
#     await start_download()


# def download() -> None:
#     asyncio.run(download_async())
