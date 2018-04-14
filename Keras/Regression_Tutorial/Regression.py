#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:22:29 2017

@author: napster
"""

import numpy as np 
import pandas as pd 
from keras.layers import Dense 
from keras.models import Sequential
from sklearn.model_selection import cross_val_score 
from keras.wrappers.scikit_learn import KerasRegressor 
from sklearn.model_selection import KFold 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 

seed = 7 
np.random.seed(seed) 

# reading dataset 
data = pd.read_csv('housing.csv', delim_whitespace=True, header=None).values 
X = data[:, 0:-1]
Y = data[:, -1]
 
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0) 
fold = KFold(n_splits=10, shuffle=True, random_state=seed) 
results = cross_val_score(estimator, X, Y, cv=fold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# creating standardized version 
estimators = [] 
estimators.append(('standardize', StandardScaler())) 
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators) 
fold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=fold) 
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# define a larger(wider/deeper) model 
def larger_model(): 
    model = Sequential()
    model.add(Dense(units=26, input_dim=13, kernel_initializer='uniform', activation='relu')) 
    model.add(Dense(units=52, kernel_initializer='uniform', activation='relu')) 
    model.add(Dense(units=26, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform')) 
    return model 

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=100, batch_size=5, verbose=0))) 
pipeline = Pipeline(estimators) 
fold = KFold(n_splits=10, shuffle=True, random_state=seed) 
results = cross_val_score(pipeline, X, Y, cv=fold) 
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std())) 






