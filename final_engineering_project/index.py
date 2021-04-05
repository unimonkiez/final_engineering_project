import torch
from .properties import data_path
from .download.main import start_download
import asyncio


def start() -> None:
    print("PyTorch version is: {0}".format(torch.__version__))  # type: ignore
    print(data_path)


async def download_async() -> None:
    await start_download()


def download() -> None:
    asyncio.run(download_async())
