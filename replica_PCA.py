#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 20:45:19 2018

@author: fernando
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp

#function to plot subplots
def plot_sub(x,t):
    """
    function to plot, in subplots, a matrix x with n signals (rows)
    """
    
    n = len(x.T)

    ax1 = plt.subplot(n,1,1)
    plt.plot(t,x[:,0],'k')
    plt.xlabel('Time [sec]')
    plt.ylabel('a.u.')   
    for i in range(2,n+1):
        plt.subplot(n,1,i,sharex = ax1)
        plt.plot(t,x[:,i-1],'k')
        plt.xlabel('Time [sec]')
        plt.ylabel('a.u.') 

#%%
#------------------------------------------------------------
#working PCA example using data from challenge
#------------------------------------------------------------

plt.close('all')
#Load signal
signals = sp.loadmat(str("a103l.mat"))

#let's define x as the matrix with rows == time  and cols== signals
x = signals['val'].T

#remove mean for each signals. We do not need it
#x_no_mean = x - np.mean(x,axis = 0, keepdims = True)
fs = 250. #sampling frequency in Hz
t = np.arange(0,len(x[:,0]))/fs
#let's plot.
plot_sub(x,t) 
    
#NOTE: we attached x axis to make the same zoom in every subplot
#%%

#Hasta aqui mas o menos bien
from sklearn.decomposition import PCA, KernelPCA

#What I removed is only particular for 12 lead ECGs. We do not need it here.

#standardize
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_std = scaler.fit_transform(x)

pca = PCA()
X_pca = pca.fit_transform(x_std)
#X_pca = pca.fit_transform(ecg_8leads)


#Caution it may take too long
kpca = KernelPCA(kernel = 'cosine',degree = 2,n_components=6,remove_zero_eig=True, gamma = 10,n_jobs = -1)
#X_kpca = kpca.fit_transform(x_std[:15000,:])
X_kpca = kpca.fit_transform(x_std[:3000,:])

#X_back = kpca.inverse_transform(X_kpca)

eigenvalues_pca = pca.explained_variance_
lambdas_kpca = kpca.lambdas_

print(X_pca.shape)
print(X_kpca.shape)

#plot eigenvectors
#%%
plt.figure()
plot_sub(X_pca,t)
plt.title('Projections onto Egeinvectors')

plt.figure()
plot_sub(X_kpca[:,:5],t[:3000])
#plt.title('Projections onto Egeinvectors - Kernel PCA')