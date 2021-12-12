from sklearn.dummy import DummyClassifier

def get_label(category):
    if cateogry == ""

def create_baseline():
    baseline = DummyClassifier(strategy="most_frequent")
    return baseline

def train_model(model,training_df):
    model = model.fit(training_df['title'].values,training_df['label'].values)
    return model

def predict_title(model,title):
    prediction = model.predict(title)

