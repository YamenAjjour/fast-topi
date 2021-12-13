import unittest
from experiment import *

class TestExperiment(unittest.TestCase):

    def create_dummy_dataframe(self):
        labels = [0, 1, 2, 3, 0]
        return pd.DataFrame({"label":labels,"title":["title" for i in range(0,5)]})

    def create_dummy_test_set(self):
        labels = np.array([0, 1, 2, 3, 0])
        titles= ["title" for i in range(0,5)]
        return Set(titles=titles,labels=labels)

    def create_dummy_classifier(self):
        dummy_classifier = DummyClassifier(strategy="constant",constant=0)

        dummy_classifier.fit(["title"],[0])
        return dummy_classifier

    def test_split_dataset(self):
        create_experiment()
        df = self.create_dummy_dataframe()
        split = split_dataset(df,holdout_perc=0.2)
        self.assertEqual(split.training.shape[0],4)
        self.assertEqual(split.test.shape[0],1)

    def test_calculate_effectiveness(self):
        test_set = self.create_dummy_test_set()
        dummy_classifier = self.create_dummy_classifier()
        prediciton = dummy_classifier.predict(["title"])
        self.assertEqual(prediciton,0)
        metrics = calculate_effectiveness(dummy_classifier,test_set)
        self.assertEqual(metrics[0].p,0.4)
        self.assertEqual(metrics[0].r,1)
        self.assertEqual(round(metrics[0].f1,2),0.57)
        self.assertEqual(metrics['macro'].p,0.1)
        self.assertEqual(metrics['macro'].r,0.25)
        self.assertEqual(round(metrics['macro'].f1,2),0.14)

    def test_cross_validate(self):
        test_set = self.create_dummy_test_set()
        dummy_classifier = self.create_dummy_classifier()
        metrics = cross_validate(dummy_classifier,test_set,5)
        entairtainment_metrics = Metrics(p=0.4,r=0.4,f1=0.4)
        zero_metrics = Metrics(p=0,r=0,f1=0)
        macro_metrics = Metrics(p=0.1,r=0.1,f1=0.1)
        dummy_average_metrics={}
        dummy_average_metrics[0]=entairtainment_metrics
        dummy_average_metrics[1]=zero_metrics
        dummy_average_metrics[2]=zero_metrics
        dummy_average_metrics[3]=zero_metrics
        dummy_average_metrics['macro']=macro_metrics
        for label in metrics:
            self.assertEqual(dummy_average_metrics[label].p,metrics[label].p)
            self.assertEqual(dummy_average_metrics[label].r,metrics[label].r)
            self.assertEqual(dummy_average_metrics[label].f1,metrics[label].f1)

