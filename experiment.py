import argparse
from collections import namedtuple, defaultdict
import pandas as pd
from config import *
from dataset import *
import os
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score, precision_score

from model import *

Split = namedtuple("Split", "test training")
Metrics = namedtuple("Metrics", "p r f1")


def split_dataset(dataset_df, holdout_perc):
    """
    Splits a given dataframe into two frames based on the holdout percentage.
    The holdout set will be sampled from the dataset using the holdout percentage while the training set will contain
    the rest of the dataset.

    :param dataset_df: dataframe to be split.
    :param holdout_perc: percentage of the holdout set.
    :return: split consisting of the holdout and the training sets.
    """
    holdout_df = dataset_df.sample(frac=holdout_perc)
    training_df = dataset_df[~dataset_df.index.isin(holdout_df.index)]
    return Split(test=holdout_df, training=training_df)


def calculate_effectiveness(model, test_set):
    """
    Calculates the precision, recall, and f1-score for the four labels and their macro-avareage.
    :param model: trained classifier model which takes strings as input and returns four labels
    :param test_set: test set to evaluate the model on
    :return: dictionary whose keys are the four labels and a 'macro' label and whose values are the
    precision, recall, and f1-score stored as Metrics
    """
    metrics = {}

    titles, labels = zip(*test_set)
    predictions = predict(model, titles)

    f1s = f1_score(labels, predictions, average=None)
    precisions = precision_score(labels, predictions, average=None)
    recalls = recall_score(labels, predictions, average=None)

    macro_f1_score = f1_score(labels, predictions, average='macro')
    macro_precision = precision_score(labels, predictions, average='macro')
    macro_recall = recall_score(labels, predictions, average='macro')

    metrics['macro'] = Metrics(p=macro_precision, r=macro_recall, f1=macro_f1_score)

    for label, f1 in enumerate(f1s):
        metrics[label] = Metrics(p=precisions[label], r=recalls[label], f1=f1)

    return metrics


def average_metrics(splits_metrics):
    """
    Averages the metrics of a given split of the dataset. The splits are assumed to be of equal size and hence their
    average is not weighted
    :param splits_metrics:
    :return: dictionary whose keys are the four labels and a 'macro' label and whose values are the
    precision, recall, and f1-score stored as Metrics
    """
    average_metric = defaultdict(lambda: 0)
    for split_metric in splits_metrics:
        for label in split_metric:
            average_metric[label] = average_metric[label] + split_metric[label]
    for label in average_metric:
        average_metric[label] = average_metric[label] / len(splits_metrics)

    return average_metric

def experiment_exists(sample=False):
    """
    Checks whether the holdout and training set of an experiment exists as files in the file system.
    :param sample: boolearn variable for whether to check the sample experiment or the main experiment
    :return: boolean variable indicating whether the experiment exists or not
    """
    path_holdout, path_training = get_pathes_experiment(sample)
    return os.path.exists(path_holdout) and os.path.exists(path_training)

def create_experiment(sample=False):
    """
    Creates a sample experiment with a training and test sets. The size of the test set depends on the holdout percentage
    In config.yaml. The sets will be stored in the configured path in config.yaml. An option for sampling the dataset can
    be set using the input parameter smpale.
    config.yaml.
    :param sample: boolearn variable for whether to create a sample experiment with the size in config.yaml
    :return: None
    """
    holdout_perc = get_holdout_percentage()
    path_holdout, path_training = get_pathes_experiment(sample)
    if sample:
        sample_experiment_size = get_sample_experiment_size()
        dataset_df = load_dataset(sample_experiment_size)
    else:
        dataset_df = load_dataset()

    split = split_dataset(dataset_df, holdout_perc)

    split.test.to_csv(path_holdout, sep="\t", encoding="utf-8")
    split.training.to_csv(path_training, sep="\t", encoding="utf-8")


def load_experiment(sample=False):
    """
    Loads the training and test sets for the experiment.
    :param sample: boolearn variable  for whether to loads the sample experiment
     or the main experiment.
    :return: a split the contains the test and training sets.
    """
    if not experiment_exists(sample):
        create_experiment(sample)
    path_holdout, path_training = get_pathes_experiment(sample)
    holdout_df = pd.read_csv(path_holdout, sep="\t", encoding="utf-8")
    training_df = pd.read_csv(path_training, sep="\t", encoding="utf-8")
    holdout = zip(holdout_df['title'].values, holdout_df['label'].values)
    training = zip(training_df['title'].values, training_df['label'].values)
    return Split(test=holdout, training=training)


def create_cross_validation_experiments(splits_count, sample=False):
    """
    Creates a cross validation experiments by splitting the training set into 5 folds.
    :param splits_count: count of splits to divide the training set into.
    :param sample:  boolean variable for whether to create a sample experiment
    :return: training and test indices for the splits together with the training set.
    """
    split = load_experiment(sample)
    training_set = split.training
    titles, labels = zip(*training_set)
    kfold = KFold(n_splits=splits_count, shuffle=True, random_state=122)
    indices = kfold.split(titles, labels)

    return indices, training_set


def cross_validate(model, splits_count, sample):
    """
    Runs a cross-validation experiments with a given split count for an input model. The training set stored in
    config.yaml will be divided into the input count of splits and the model will be trained and evaluated on
    each split.
    :param model: model to be evaluated.
    :param splits_count: count of splits to divide the trianing set into
    :param sample: whether to use the training set of the sample experiment
    :return: average precision, recall, and f1_score for each label and the macro average after averaging them
    on all splits.
    """

    indices, training_set = create_cross_validation_experiments(splits_count, sample)
    titles, labels = zip(*training_set)
    all_folds_metrics = []
    for training_index, test_index in indices:
        title_training, titles_test = titles[training_index], titles[test_index]
        labels_training, labels_test = labels[training_index], labels[test_index]
        training_fold_set = zip(title_training, labels_training)
        title_fold_set = zip(titles_test, labels_test)
        train_model(model, training_fold_set)
        metrics = calculate_effectiveness(model, title_fold_set)
        all_folds_metrics.append(metrics)
    averaged_metrics = average_metrics(all_folds_metrics)
    return averaged_metrics

def test(model,split):
    """
    Run an experiment on the holdout set after training it on the training set.

    :param model: model to be evaluated.
    :param split: split consisting of the holdout and the training sets
    :return: average precision, recall, and f1_score for each label and the macro average
    """
    training_set = split.training
    test_set = split.test
    train_model(model,training_set)
    metrics = calculate_effectiveness(model,test_set)
    return metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action="store_true", help="run the experiment using the baseline")
    parser.add_argument("--train", action="store_true", help="train the model and store it")
    parser.add_argument("--sample", action="store_true", help="use the sample dataset")
    parser.add_argument("--crossvalidate", action="store_true", help="run a cross validation experiment")
    args = parser.parse_args()
    return args


def validate_args(args):
    pass


def main():
    args = parse_args()
    validate_args(args)
    arg_baseline = args.baseline
    arg_train = args.train
    arg_sample = args.sample
    arg_cross_validate = args.crossvalidate

    if arg_baseline and arg_train and arg_sample:
        model = create_baseline()
        sample_size = get_sample_experiment_size()
        sample = load_dataset(sample_size, zip_dataset=True)
        train_model(model, sample)
        path_model = get_path_model("baseline")
        dump_model(model, path_model)

    if arg_baseline and arg_cross_validate and arg_sample:
        model = create_baseline()
        splits_count = get_splits_count()
        metrics = cross_validate(model,splits_count,arg_sample)
        print(metrics)



if __name__ == '__main__':
    main()
