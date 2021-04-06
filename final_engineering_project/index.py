from final_engineering_project.data.main import create_data

# import torch
# import asyncio


def start() -> None:
    # print("PyTorch version is: {0}".format(torch.__version__))  # type: ignore
    create_data()


# async def download_async() -> None:
#     await start_download()


# def download() -> None:
#     asyncio.run(download_async())
