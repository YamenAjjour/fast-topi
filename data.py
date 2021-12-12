import pandas as pd

def preprocess_dataset():
    pass
def get_label(category):
    if category == "e":
        return 0
    elif category =="b":
        return 1
    elif category =="t":
        return 2
    elif category =="m":
        return 3
    else:
        raise ValueError("Unrecognized label")

def get_descriptive_category(label):
    if label==0:
        return "entertainment"
    elif label==1:
        return "business"
    elif label==2:
        return "tech"
    elif label==3:
        return "health"
    else:
        raise ValueError("Unrecognized label")