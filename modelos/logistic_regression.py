#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:59:02 2018
@author: fernando
"""

import itertools
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, \
                            f1_score, log_loss, precision_score, \
                            recall_score, roc_auc_score, confusion_matrix

X = np.load("../Data.npy")
Y = np.load("../Output.npy")
X = np.delete(X, (380), axis=0)
Y = np.delete(Y, (380), axis=0)

model = LogisticRegression()
X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=0.15, random_state=0)
        #Downsampling training set


model.fit(X_train,y_train)
print model

#train and save this model
y_lr = model.predict(X_test)
np.save('y_lr.npy', y_lr)
