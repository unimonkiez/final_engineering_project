from os import environ
from dotenv import load_dotenv

load_dotenv(dotenv_path="./final_engineering_project/properties.list")

kaggle_path = environ["KAGGLE_PATH"]
noise_path = environ["NOISE_PATH"]
train_path = environ["TRAIN_PATH"]
test_path = environ["TEST_PATH"]
model_path = environ["MODEL_PATH"]
optimizer_path = environ["OPTIMIZER_PATH"]
