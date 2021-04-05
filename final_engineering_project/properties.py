from os import environ
from dotenv import load_dotenv

load_dotenv(dotenv_path="./final_engineering_project/properties.list")

data_path = environ["DATA_PATH"]
kaggle_relative_path = environ["KAGGLE_RELATIVE_PATH"]
