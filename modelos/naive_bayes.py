#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:25:06 2018
@author: fernando
"""

from sklearn.naive_bayes import GaussianNB
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, \
                            f1_score, log_loss, precision_score, \
                            recall_score, roc_auc_score, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

X = np.load("Data.npy")
Y = np.load("Output.npy")
X = np.delete(X, (380), axis=0)
Y = np.delete(Y, (380), axis=0)
model = GaussianNB()#load model
#put this parameters into my model
X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=0.15, random_state=0)
        #Downsampling training set


model.fit(X_train,y_train)
print model
#evalate model on training set
y_nb = model.predict(X_test)
np.save('y_lr.npy', y_nb)

y_prob_nb = model.predict_proba(X_test)
np.save('y_prb_nb', y_prob_nb)
