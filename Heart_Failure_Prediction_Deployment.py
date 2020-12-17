#Importing the libraries required

import numpy as np
import pandas as pd
import pickle

# Loading the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Dropping the columns which has no or very less connection with the output
data.drop(['diabetes','sex','smoking','anaemia','serum_sodium'],axis=1,inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = data.drop(columns='DEATH_EVENT')
y = data['DEATH_EVENT']

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=120, max_features='log2', min_samples_leaf=2,
                       min_samples_split=3, n_estimators=200)
classifier.fit(X, y)

# Creating a pickle file for the classifier
filename = 'heart_failure_prediction_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
