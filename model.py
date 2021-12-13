import pickle
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset import *

def create_baseline():
    baseline = DummyClassifier(strategy="constant",constant=[0])
    return baseline

def create_model():
    token_n_grams = CountVectorizer(ngram_range=(1,3))
    pipeline = Pipeline([('ngram',CountVectorizer),('normalizer',StandardScaler(with_mean=False)),('logistic-regression',logistic_regression)])

def train_model(model,training_set):
    titles,labels =  zip(*training_set)
    model = model.fit(titles,labels)
    return model

def predict_title(model,title):
    prediction = model.predict(title)
    category = get_descriptive_category(prediction[0])
    return category

def predict(model,titles):
    return model.predict(titles)

def dump_model(model, path_model):
    with  open(path_model, 'wb') as model_file:
        pickle.dump(model,model_file)

def load_model(path_model):
    with open(path_model,'rb') as model_file:
        model= pickle.load(model_file)
        return model

