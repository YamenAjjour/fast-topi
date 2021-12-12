import argparse
from collections import namedtuple
import pandas as pd

from config import *
from dataset import *
from model import *



Split = namedtuple("Split", "test training")


def split_dataset(dataset_df,holdout_perc):

    holdout_df = dataset_df.sample(holdout_perc)
    training_df = dataset_df[~dataset_df.index.isin(holdout_df.index)]
    return Split(test=holdout_df,training=training_df)

def create_sample_experiment():
    """
    Create a sample experiment with a training and test splits
    :return: A Split
    """
    path_holdout, path_training = get_pathes_sample_experiment()
    sample_experiment_size = get_sample_experiment_size()
    holdout_perc = get_holdout_percentage()
    dataset_df = load_dataset()
    sample_df = dataset_df.sample(sample_experiment_size)
    split =split_dataset(sample_df,holdout_perc)

    split.test.to_csv(path_holdout,sep="\t",encoding="utf-8")
    split.training.to_csv(path_training,sep="\t",encoding="utf-8")

def load_sample_experiment():

    path_holdout, path_training = get_pathes_sample_experiment()
    holdout_df = pd.read_csv(path_holdout,sep="\t",encoding="utf-8")
    training_df = pd.read_sas(path_training,sep="\t",encoding="utf-8")

    return Split(test=holdout_df,training=training_df)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline',action="store_true",help="run the experiment using the baseline")
    parser.add_argument("--train",action="store_true",help="train the model and store it")
    parser.add_argument("--sample",action="store_true",help="use the sample dataset")
    args=parser.parse_args()
    return args

def validate_args(args):
    pass

def main():

    args= parse_args()
    validate_args(args)
    use_baseline = args.baseline
    train = args.train
    sample = args.sample
    if use_baseline and train and sample:
        baseline=create_baseline()
        sample_df=load_sample()
        train_model(baseline,sample_df)
        path_model=get_path_model("baseline")
        dump_model(baseline,path_model)


if __name__ == '__main__':
    main()


