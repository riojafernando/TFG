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

#StandarScaler normalize features by removing the mean and scaling to unit variance
scaler = StandardScaler() 
x_std = scaler.fit_transform(x)
#windowed signals
x_windowed = {}

for i in range(5):
    x_windowed[i] = x_std[i*16500:16500*(i+1)]

#Make PCA for each section
pca = PCA()
X_pca_w = {}
X_kpca_w = {}
eigenvalues_pca = {}
lambdas_kpca = {}
kpca = KernelPCA(kernel = 'cosine',degree = 2,n_components=10,remove_zero_eig=True, gamma = 10,n_jobs = -1)
#X_kpca = kpca.fit_transform(x_std[:15000,:])
for i in range(5):
    
    X_pca_w[i] = pca.fit_transform(x_windowed[i])
    eigenvalues_pca[i] = pca.explained_variance_
    X_kpca_w[i] = kpca.fit_transform(x_windowed[i][0:2000])
    #Con todas las señales muere
    lambdas_kpca[i] = kpca.lambdas_
    print i
#X_pca = pca.fit_transform(ecg_8leads)

#X_kpca_w[1] = kpca.fit_transform(x_windowed[1][0:2000])
#lambdas_kpca[1] = kpca.lambdas_
print ("termino KPCA")

#Caution it may take too long


#X_back = kpca.inverse_transform(X_kpca)
    
#print(X_pca.shape)
#print(X_kpca.shape)
#plot eigenvectors
#%%
plt.figure()
#plot_sub(X_pca,t)
plt.title('Projections onto Egeinvectors')

plt.figure()
plot_sub(X_kpca[:,:5],t[:3000])
#plt.title('Projections onto Egeinvectors - Kernel PCA')
