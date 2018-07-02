# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 18:25:39 2018

@author: Fernando
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, \
                            f1_score, log_loss, precision_score, \
                            recall_score, roc_auc_score, confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



######### XGBOOST #############
print("- Xgboost -")
print("===========")
y_test = np.load('y_test.npy')
y_xgb = np.load('y_xgb.npy')
y_prob_xgb = np.load('y_prob_xgb.npy')

accuracy = accuracy_score(y_test, y_xgb)
print("accuracy_score: ",str(accuracy))

precision = precision_score(y_test, y_xgb)
print("precision_score: ",str(precision))

f1 = f1_score(y_test, y_xgb)
print ("f1: ",str(f1))

recall = recall_score(y_test, y_xgb)
print("recall_score: " ,str(recall))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_xgb)
# Plot non-normalized confusion matrix
plt.figure()
class_names = ['True Alarm', 'False Alarm']
class_names = np.array(class_names)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix Xgboost, without normalization')
print("- Naive Bayes -")
print("===============")
######### NAIVE BAYES #############
y_nb = np.load('y_nb.npy')
y_prob_bn = np.load('y_prob_nb.npy')
accuracy = accuracy_score(y_test, y_nb)
print("accuracy_score: ",str(accuracy))

precision = precision_score(y_test, y_nb)
print("precision_score: ",str(precision))

f1 = f1_score(y_test, y_nb)
print("f1: " ,str(f1))

recall = recall_score(y_test, y_nb)
print("recall_score: ",str(recall))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_nb)
# Plot non-normalized confusion matrix
plt.figure()
class_names = ['True Alarm', 'False Alarm']
class_names = np.array(class_names)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix naive Bayes, without normalization')

####### LOGISTIC REGRESION #########
print("- Logistic Regression -")
print("=======================")
y_lr = np.load('y_lr.npy')
accuracy = accuracy_score(y_test, y_lr)
print("accuracy_score: " ,str(accuracy))

precision = precision_score(y_test, y_lr)
print("precision_score: ",str(precision))

f1 = f1_score(y_test, y_lr)
print ("f1: ", str(f1))

recall = recall_score(y_test, y_lr)
print ("recall_score: ",str(recall))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_lr)
# Plot non-normalized confusion matrix
plt.figure()
class_names = ['True Alarm', 'False Alarm']
class_names = np.array(class_names)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix logistic regression, without normalization')

####### COMBINE MODEL ####################
print("- Combine Model -")
print("===============")
y_combine = np.load('y_combine.npy')
accuracy = accuracy_score(y_test, y_combine)
print("accuracy_score: ",str(accuracy))

precision = precision_score(y_test, y_combine)
print("precision_score: ",str(precision))

f1 = f1_score(y_test, y_combine)
print("f1: ",str(f1))

recall = recall_score(y_test, y_combine)
print("recall_score: ",str(recall))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_combine)
# Plot non-normalized confusion matrix
plt.figure()
class_names = ['True Alarm', 'False Alarm']
class_names = np.array(class_names)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix combine model, without normalization')
