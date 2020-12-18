#Importing the libraries required

import numpy as np
import pandas as pd
import pickle

# Loading the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Dropping the columns which has no or very less connection with the output
data.drop(['anaemia','creatinine_phosphokinase'],axis=1,inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = data.drop(columns='DEATH_EVENT')
y = data['DEATH_EVENT']

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(criterion='gini', max_depth=340, max_features='sqrt',
                       min_samples_split=5, n_estimators=1400)
classifier.fit(X,y)

# Creating a pickle file for the classifier
filename = 'heart_failure_prediction_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
