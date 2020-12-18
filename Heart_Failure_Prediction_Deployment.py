#Importing the libraries required

import numpy as np
import pandas as pd
import pickle

# Loading the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Dropping the columns which has no or very less connection with the output
data.drop(['anaemia'],axis=1,inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
X = data.drop(columns='DEATH_EVENT')
y = data['DEATH_EVENT']

# Creating XGBoost Model
from xgboost import XGBClassifier
classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.6, gamma=0.1, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=3,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
classifier.fit(X,y)

# Creating a pickle file for the classifier
filename = 'heart_failure_prediction_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
