from final_engineering_project.properties import data_path, kaggle_relative_path
from final_engineering_project.tool.run import run

download_path = "{absolute_path}/{relative_path}".format(
    absolute_path=data_path,
    relative_path=kaggle_relative_path,
)


async def download_kaggle() -> None:
    print(
        "Downloading kaggle to {download_path}...".format(
            download_path=download_path,
        ),
    )

    await run(
        "kaggle competitions download -c freesound-audio-tagging -p {download_path}".format(
            download_path=download_path,
        ),
    )
