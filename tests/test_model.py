import unittest
import os.path
from config import *
from model import *

class TestModel(unittest.TestCase):
    def test_dump_model(self):
        try:
            baseline = create_baseline()
            path=get_path_model("baseline-testing")
            if os.path.exists(path):
                os.remove(path)
            dump_model(baseline,path)
            self.assertTrue(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.remove(path)
    def create_dummy_test_set(self):
        labels = np.array([0, 1, 2, 3, 0])
        titles= ["title" for i in range(0,5)]
        return Set(titles=titles,labels=labels)

    def test_load_baseline(self):
        try:
            training_set = self.create_dummy_test_set()
            baseline = create_baseline()
            train_model(baseline,training_set)
            path=get_path_model("baseline-testing")
            dump_model(baseline,path)
            baseline = load_model(path)
            prediction = predict_title(baseline,"title")
            self.assertEqual(prediction,"entertainment")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_load_model(self):
        path_model = get_path_model("model")
        if os.path.exists(path_model):
            model = load_model(path_model)
            label = predict_title(model,"'entertainment news title")
            self.assertIn(label, ["tech", "entertainment", "health", "business"])

