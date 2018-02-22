#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:01:55 2018

@author: obarquero
"""

import numpy as np
import matplotlib.pyplot as plt

#ejemplo PCA
ecg = np.loadtxt("./1698287.txt",delimiter = ',') #acceptable
#ecg = np.loadtxt("./files/1568146.txt",delimiter = ',') #unacceptable
#ecg = np.loadtxt("./files/1101829.txt",delimiter = ',') #unacceptable
ecg = ecg - np.mean(ecg,axis = 0)
accep = True
if accep == True:
    for n in range(12):
        plt.plot(ecg[:,n+1]+n*400,color = 'k')
else:
    for n in range(12):
        plt.plot(ecg[:,n+1]+n*200,color = 'k') 

plt.ylim((-10,2500))


from sklearn.decomposition import PCA, KernelPCA

#we need to remove leads which are linear combination of the remaining
ecg_12 = ecg[:,1:]
print np.shape(ecg_12)
ecg_8leads = np.concatenate((ecg_12[:,0:2],ecg_12[:,6:]),axis = 1)
print np.shape(ecg_8leads)

#kpca = KernelPCA(kernel = 'rbf', fit_inverse_transform =True, gamma = 10)
#X_kpca = kpca.fit_transform(ecg[:,1:])
#X_back = kpca.inverse_transform(X_kpca)
pca = PCA()
X_pca = pca.fit_transform(ecg_8leads)


print(X_pca.shape)

plt.figure(figsize = (15,5))

#make different plots wheter is acc or unacc

if accep == False:
    
    plt.subplot(121)
    for n in range(8):
        plt.plot(ecg_8leads[:,n]+n*200,color = 'k') 
        plt.ylim((-150,1500))
        plt.axis('off')
    plt.subplot(122)
    for m in range(8):
        plt.plot(X_pca[:,7-m]+m*450,color = 'k')
    plt.ylim((-30,3500))
    plt.axis('off')
    
else:
    plt.subplot(121)
    for n in range(8):
        plt.plot(ecg_8leads[:,n]+n*350,color = 'k') 
        plt.ylim((-150,2700))
        plt.axis('off')
    plt.subplot(122)
    for m in range(8):
        plt.plot(X_pca[:,7-m]+m*690,color = 'k')
    plt.ylim((-30,5500))
    plt.axis('off')
    
#Plot the eigenvalues
plt.figure()
plt.plot(pca.explained_variance_/np.sum(pca.explained_variance_),linewidth = 2,marker = 'o')
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('Explained_variance')

pca.explained_variance_
        