# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:51:16 2018
@author: Fernando
"""
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#combine methods

#cargamos los modelos de naive bayes y de xgboost
y_nb = np.load('y_nb2.npy')
y_xgb = np.load('y_xgb.npy')

y_prob_nb = np.load('y_prob_nb.npy')
y_prob_xgb = np.load('y_prob_xgb.npy')

combine = np.zeros(len(y_nb))


for i in range(len(y_nb)):
    #comprobar que son iguales, 0, 0
    if y_nb[i] == 0 and y_xgb[i] == 0:
        combine[i] = 0 #elección por votación
    #comprobar que son iguales 1,1
    elif y_nb[i] == 1 and y_xgb[i] == 1:
        combine[i] = 1 #elección por votación
       
    #si son diferentes
    elif y_nb[i] == 1 and y_xgb[i] == 0:
        
        #si la probabilida de uno
        if y_prob_nb[i,1] > y_prob_xgb[i,0]:
            combine[i] = 1
        elif y_prob_xgb[i,0] > y_prob_nb[i,1]:
            combine[i] = 0
            
    elif y_nb[i] == 0 and y_xgb[i] == 1:
        if y_prob_xgb[i,1] > y_prob_nb[i,0]:
            combine[i] = 1
        elif y_prob_nb[i,0] > y_prob_xgb[i,1]:
            combine[i] = 0
            
            
y_combine = combine

np.save('y_combine.npy', y_combine)
