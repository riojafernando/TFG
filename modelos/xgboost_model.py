#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 18:19:49 2018

@author: fernando
"""

from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot
import pickle

#load data
X = np.load("../Data.npy");
Y = np.load("../Output.npy")
#X = np.load(data)
#Y = np.load(output)


#Aqui faltaria hacer tambien la separacion entre train y test
X_train, X_test, y_train, y_test = \
            train_test_split(X, Y, test_size=0.15, random_state=0)

#Statrting with xgboost parametres
model = XGBClassifier() #our model of xgboost

#parameters that I try to modified
n_estimators = list(range(20, 400, 30))
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
max_depth = [2, 4, 6, 8]

#put this parameters into my model
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)

#specificate cross-validation
kfold = StratifiedKFold(n_splits=8, shuffle=True, random_state=7)

#training our model
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold,verbose = 1)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

pickle.dump(grid_result, open("xgboost_CV.pickle.dat", "wb"))
print("Saved model to: pima.pickle.dat")
# some time later...
# load model from file
grid_result = pickle.load(open("xgboost_CV.pickle.dat", "rb"))

#Con el mejor resultado, crear un modelo xgboost que tenga los hiperparametros del mejor resultado
#luego probar los resultados en test y comparar contra naive bayes y regresion logistica

#---------------------#---------------------#---------------------#---------------------#---------------------
#DE AQUI HACIA ABAJO SIN HABER PROBADO EN DETALLE. LO TIENE QUE EJECUTAR Y DEPURAR TU------
#---------------------#---------------------#---------------------#---------------------#---------------------
m_d = grid_result.best_params_['max_depth']
l_r = grid_result.best_params_['learning_rate']
n_e = grid_result.best_params_['n_estimators']

best_model = XGBClassifier(max_depth =m_d,learning_rate = l_r,n_estimators = n_e)

#reentrenamos con todo el conjunto de training

best_model.fit(X_train,y_train)

#ahora ya podemos sacar métricas.

y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)

#vamos a sacar también información de la curva ROC y también vamos a ver como cambia todo moviendo los thresholds
#esto te lo explico en persona la semana que viene mejor.

#todo el resumen de la matriz de confusión y demás.
print(confusion_matrix(y_test,y_pred))