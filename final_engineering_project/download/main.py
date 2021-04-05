from .kaggle import download_kaggle


async def start_download() -> None:
    print("Downloading...")
    await download_kaggle()
    print("Done!")
