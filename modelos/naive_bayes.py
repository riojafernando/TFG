#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:25:06 2018

@author: fernando
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

data = open("Data.npy");
output = open("Output.npy")
X = np.load(data)
Y = np.load(output)
model = GaussianNB()#load model
#put this parameters into my model
X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=0.15, random_state=0)
        #Downsampling training set


model.fit(X_train,y_train)
print model

y_pred_nb = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_nb)
print "accuracy_score: " + str(accuracy)
