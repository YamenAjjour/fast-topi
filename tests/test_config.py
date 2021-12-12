import config
import unittest
from config import *
class TestConfig(unittest.TestCase):
    def get_path_root(self):
        return  abspath(os.path.dirname(__file__) + "/../")


    def test_get_pathes_sample_experiment(self):
        path_holdout, path_training = config.get_pathes_sample_experiment()
        self.assertEqual(path_holdout,"data/holdout-sample.csv")
        self.assertEqual(path_training,"data/training-sample.csv")

    def test_path_source_dataset(self):
        path_source_dataset = get_path_source_dataset()
        self.assertEqual(path_source_dataset,"data/newsCorpora.csv")

    def test_path_preprocessed_dataset(self):
        path_preprocessed_dataset = get_path_preprocessed_dataset()
        self.assertEqual(path_preprocessed_dataset,"data/dataset-preprocessed.csv")


