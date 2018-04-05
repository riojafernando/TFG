#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 20:38:38 2018

@author: fernando
"""

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load data
#8 patient features --> 1 output, means that patient will have diabetes in 5 years
dataset = loadtxt('prueba.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets with a seed for we saw the same results
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)

#train our XgboostModel
# fit model on training data


model = XGBClassifier()
model.fit(X_train, y_train, param0)
print(model)
