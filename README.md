
*Fast-topi* is a framework to develop and deploy models as an api service. The model or the service takes as an input a title 
and returns one of the following categories: **Entertainment**, **Tech**, **Business**, and **Health**. 
The framework is centered around a common configuration file ``config.yaml`` where all parameters for developing the model
and deploying the REST api are stored. 

The general use case for using *fast-topi* is deploying a service and then querying it via the command line.
```
./deploy.sh
python client.py --title Elon Musk named Time's Person of the year for 2020
```
## Install
To install fast-topi using pip run
```
pip install fast-topi
```
## REST API Deployment
To deploy a REST api locally run the following command. 
```
./deploy.sh
```

Once the REST api is up and running, you can use the following path to query it
```
http://127.0.0.1:80/categories/?title=Elon usk named Time's Person of the year for 2021
```
This will retrun a json object containing {"category":"tech"} for this title. 
## Command line client 
To get the news category for a given title you can use the command line client.
A prerequisite is that the rest api is already deploied. 
```
python client.py --title "Elon usk named Time's Person of the year for 2021"
```
 

## Experiments
To train and test a new model you can run one of the following experiments. By default, a logistic regression model
will be used using token n-grams. An alterantive is using a majority baseline by using ``--baseline``.  


#### Cross validation experiments
To run a n-folds cross validation experiments on part of the dataset run the following command. This will create 
a holdout set on which can be used to evaluate the model. The effectiveness of the classifier will be calculated for 
all available hyper parameter (c) in the ``config.yaml`` file. By default, the classifier will be evaluated on 5
splits and a holdout set will be created with 10 % of the whole dataset. To change these parameters you can edit the 
parameters ``split_counts `` and ``holdout_perc`` in the configuration file ``config.yaml``. 

```
python experiment.py --crossvalidate 
```

#### Testing on holdout set
Runs a one-split experiment classifier on the dataset by creating a holdout set which will be used to evaluate the model. 
```
python experiment.py --test 
```

#### Training a model
To train a final model on the whole dataset and use it for the REST api use the command. This will store a new model
under ``models/model.pkl``. To change the default path of the model, edit the ``config.yaml`` file. 
```
python experiment.py --train 
```

The experiment script allows to run experiments on a sample of the dataset using ``--sample``. The size of the sample 
is stored on the ``config.yaml`` file. 
## Code testing
```
 python -m unittest tests/*.py
```

## Configuration
The configuration for the REST api, model, experiments, and dataset are stored as yaml file under ``config.yaml``.


## Dependencies
To install the needed dependencies use the following command.

```pip install --r requirements.txt``` 
