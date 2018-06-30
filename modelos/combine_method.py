# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:51:16 2018

@author: Fernando
"""
import numpy as np
#combine methods

y_nb = np.load('y_nb2.npy')
y_xgb = np.load('y_xgb.npy')

y_prob_nb = np.load('y_prob_nb.npy')
y_prob_xgb = np.load('y_prob_xgb.npy')

combine = np.zeros(len(y_nb))

for i in range(71):
    if y_nb[i] == 0 and y_xgb[i] == 0:
        combine[i] = 0
    elif y_nb[i] == 1 and y_xgb[i] == 1:
        combine[i] = 1
    elif y_nb[i] == 1 and y_xgb[i] == 0:
        if y_prob_nb[i,1] > y_prob_xgb[i,0]:
            combine[i] = 1
        elif y_prob_xgb[i,0] > y_prob_nb[i,1]:
            combine[i] = 0
    elif y_nb[i] == 0 and y_xgb[i] == 1:
        if y_prob_xgb[i,1] > y_prob_nb[i,0]:
            combine[i] = 1
        elif y_prob_nb[i,0] > y_prob_xgb[i,1]:
            combine[i] = 0
            
