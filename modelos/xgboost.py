#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:19:49 2018

@author: fernando
"""

from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot

#load data
data = open("Data.npy");
output = open("Output.npy")
X = np.load(data)
Y = np.load(output)

#Statrting with xgboost parametres
model = XGBClassifier() #our model of xgboost

#parameters that I try to modified
n_estimators = range(20, 201, 30)
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
max_depth = [2, 4, 6, 8]

#put this parameters into my model
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)

#specificate cross-validation
kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=7)

#training our model
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

