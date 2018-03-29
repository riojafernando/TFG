#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 20:45:19 2018

@author: fernando
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
import os

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
def extract_header():
    """
    function to print number of signals that we have from the patient
    """
    count4 = 0;
    count1_I = 0;
    count1_II = 0;
    count1_III = 0;
    count2_V = 0;
    count2_avf = 0;
    count2_III = 0;
    count2_II = 0;
    count3 = 0;
    count_ABP = 0;
    count_pleth = 0;
    count_resp = 0;
    count_ABP4 = 0;
    count_resp4 = 0;
    count2_MCL = 0;
    
    #path = "/home/fernando/Escritorio/TFG/dataset/" #BE CAREFUL
    path = "./dataset/" #BE CAREFUL
    files = os.listdir(path)
    for header in files:
        if 'hea' in str(header):
            archivo = open(path + header, 'rwx')
            lineas = archivo.readlines()
            linea0 = lineas[0].split();
            linea1 = lineas[1].split();
            linea2 = lineas[2].split();
            linea3 = lineas[3].split();
            signal1 = linea1[-1]
            signal2 = linea2[-1]
            signal3 = linea3[-1]
            if '4' in linea0[1]: #look for patients with 4 signals
                linea4 = lineas[4].split();
                signal4 = linea4[-1]
                count4 += 1
                if signal4 == "ABP":#type of signal
                    count_ABP4 += 1;
                elif signal4 == "RESP":
                    count_resp4 += 1;
                else:
                    print signal4
            else:
                count3 += 1;
                

            if signal1 == "I":#same with other signals
                count1_I += 1
            elif signal1 == "II":
                count1_II += 1;
            elif signal1 == "III":
                count1_III += 1;

            if signal2 == "V":
                count2_V += 1;
            elif signal2 == "aVF":
                count2_avf = count2_avf + 1;
            elif signal2 == "III":
                count2_III += 1;
            elif signal2 == "II":
                count2_II += 1;
            elif signal2 == "MCL":
                count2_MCL += 1;

            if signal3 == "ABP":
                count_ABP += 1; 
            elif signal3 == "RESP":
                count_resp += 1;
            elif signal3 == "PLETH":
                count_pleth += 1;
            
    print ("Total de pacientes:", count3 + count4)
    print ("Total de pacientes con 3:", count3)
    print ("Total de pacientes con 4:", count4)
    print ("========================================")
    print ("Total de pacientes cuya primera es I:", count1_I)
    print ("Total de pacientes cuya primera es II:", count1_II)
    print ("Total de pacientes cuya primera es III:", count1_III)
    print ("Total de pacientes cuya primera es aVF:", 1)
    print ("========================================")
    print ("Total de pacientes cuya segunda es V:", count2_V)
    print ("Total de pacientes cuya segunda es aVF:", count2_avf)
    print ("Total de pacientes cuya segunda es II:", count2_II)
    print ("Total de pacientes cuya segunda es III:", count2_III)
    print ("Total de pacientes cuya segunda es MCL:", count2_MCL)
    print ("Total de pacientes cuya segunda es aVR:", 3)
    print ("Total de pacientes cuya segunda es aVL:", 2)
    print ("========================================")
    print ("Total de pacientes cuya tercera es ABP:", count_ABP)
    print ("Total de pacientes cuya tercera es PLETH:", count_pleth)
    print ("========================================")
    print ("Total de pacientes con cuarta que es ABP:", count_ABP4)
    print ("Total de pacientes con cuarta que es PLETH:", count_resp4)
    
    
                  
    
#%%
#------------------------------------------------------------
#working PCA example using data from challenge
#------------------------------------------------------------

plt.close('all')
#Load signal
signals = sp.loadmat(str("ejemplo4.mat"))

#let's define x as the matrix with rows == time  and cols== signals
x = signals['val'].T

#remove mean for each signals. We do not need it
#x_no_mean = x - np.mean(x,axis = 0, keepdims = True)
fs = 250. #sampling frequency in Hz
t = np.arange(0,len(x[:,0]))/fs
#let's plot.
plot_sub(x,t) 
extract_header()
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
x_windowed = []

for i in range(5):
    x_windowed.append(x_std[i*16500:16500*(i+1)])

#Make PCA for each section
pca = PCA()
X_pca_w = []
X_kpca_w = []
eigenvalues_pca = []
lambdas_kpca = []
kpca = KernelPCA(kernel = 'cosine',degree = 2,n_components=6,remove_zero_eig=True, gamma = 10,n_jobs = -1)
i = 0
#X_kpca = kpca.fit_transform(x_std[:15000,:])
for x_win in x_windowed:
    
    X_pca_w.append(pca.fit_transform(x_win))
    eigenvalues_pca.append(pca.explained_variance_)
    X_kpca_w.append(kpca.fit_transform(x_win))
    #Con todas las señales muere
    lambdas_kpca.append(kpca.lambdas_)
    print i,' de ',len(x_windowed)
    i +=1
#X_pca = pca.fit_transform(ecg_8leads)

#X_kpca_w[1] = kpca.fit_transform(x_windowed[1][0:2000])
#lambdas_kpca[1] = kpca.lambdas_
print ("termino KPCA")

#Caution it may take too long
#<<<<<<< HEAD
#kpca = KernelPCA(kernel = 'sigmoid',degree = 2,n_components=10,remove_zero_eig=True, gamma = .3,n_jobs = -1)
#X_kpca = kpca.fit_transform(x_std[:15000,:])
#X_kpca = kpca.fit_transform(x_std[:3000,:])

#X_back = kpca.inverse_transform(X_kpca)

#eigenvalues_pca = pca.explained_variance_
#lambdas_kpca = kpca.lambdas_


#X_back = kpca.inverse_transform(X_kpca)
    
#print(X_pca.shape)
#print(X_kpca.shape)
#plot eigenvectors
#%%
plt.figure()
#plot_sub(X_pca,t)
plt.title('Projections onto Egeinvectors')

plt.figure()
plot_sub(X_kpca_w[0][:,:5],t[:2000])
#plt.title('Projections onto Egeinvectors - Kernel PCA')
