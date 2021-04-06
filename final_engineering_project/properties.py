from os import environ
from dotenv import load_dotenv

load_dotenv(dotenv_path="./final_engineering_project/properties.list")

kaggle_path = environ["KAGGLE_PATH"]
noise_path = environ["NOISE_PATH"]
