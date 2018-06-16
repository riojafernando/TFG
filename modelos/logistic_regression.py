#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:59:02 2018

@author: fernando
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = open("Data.npy");
output = open("Output.npy")
X = np.load(data)
Y = np.load(output)

model = LogisticRegression()
X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=0.15, random_state=0)
        #Downsampling training set


model.fit(X_train,y_train)
print model

y_pred_nb = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_nb)
print "accuracy_score: " + str(accuracy)
