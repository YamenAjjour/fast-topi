import pickle
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset import *

def create_baseline():
    baseline = DummyClassifier(strategy="constant",constant=[0])
    return baseline

def create_model(c):
    token_n_grams = CountVectorizer(ngram_range=(1,3),max_features=1000,stop_words={'english'})
    logistic_regression = LogisticRegression(C=c)
    pipeline = Pipeline([('ngram',token_n_grams),('normalizer',StandardScaler(with_mean=False)),('logistic-regression',logistic_regression)])
    return  pipeline

def train_model(model,training_set):
    titles, labels =  training_set.titles, training_set.labels
    model = model.fit(titles,labels)
    return model

def predict_title(model,title):
    prediction = model.predict([title])
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

