from settings import train, validate, models
from utils import read_files, create_folders
from model import train_model, validate_model

if __name__ == "__main__":

    df = read_files()
    create_folders()

    if train: train_model(df, models)
    if validate: validate_model(df, models)
