import yaml
import os
from os.path import abspath

def load_config():
    with open("config.yaml") as file:
        config= yaml.safe_load(file)
        return config

def get_path_dataset_dir():
    config = load_config()
    path_dataset = config["dataset"]["path_dir"]
    return path_dataset

def get_path_preprocessed_dataset():
    config = load_config()
    path_dataset_preprocessed = config["dataset"]["path_preprocessed"]
    return path_dataset_preprocessed

def get_path_source_dataset():
    config = load_config()
    path_source_dataset =  config["dataset"]["path_source"]
    return path_source_dataset

def get_path_zip_dataset():
    config = load_config()
    path_zip_dataset = config["dataset"]["path_zip"]
    return path_zip_dataset


def get_pathes_experiment_sample():
    config = load_config()
    path_holdout = config["experiment-sample"]["path_holdout"]
    path_training =  config["experiment-sample"]["path_training"]
    return path_holdout, path_training

def get_url_dataset():
    config = load_config()
    url_dataset = config["dataset"]["url"]
    return url_dataset
