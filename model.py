import pickle
from sklearn.dummy import DummyClassifier
from dataset import *

def create_baseline():
    baseline = DummyClassifier(strategy="most_frequent")
    return baseline

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

