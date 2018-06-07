#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 19:37:11 2018

@author: fernando
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


path = "/home/fernando/Escritorio/TFG/dataset/"
files = os.listdir(path)
x_4 = {}
x_4_short = {}
x_long_windowed = [0,0,0,0,0]
x_short_windowed = [0] * 4940

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
def find_signals():
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
    total = 0;
    i = 0;
    j = 0;
    
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
            if '4' in linea0[1] and '82500' in linea0[-1] and 'b' in linea0[0]: #look for patients with 4 signals
                linea4 = lineas[4].split();
                signal4 = linea4[-1]
                count4 += 1
                mat_name = path + header[0:5] + '.mat'
                print mat_name
                archivo_mat = sp.loadmat(str(mat_name))
                x_4[i] = archivo_mat['val'].T
                if signal4 == "ABP":#type of signal
                    count_ABP4 += 1;
                elif signal4 == "RESP":
                    count_resp4 += 1;
                    
                i += 1; 
                 
            elif '4' in linea0[1] and '75000' in linea0[-1]:
                linea4 = lineas[4].split();
                signal4 = linea4[-1]
                count4 += 1
                mat_name = path + header[0:5] + '.mat'
                archivo_mat = sp.loadmat(str(mat_name))
                x_4_short[j] = archivo_mat['val'].T
                if signal4 == "ABP":#type of signal
                    count_ABP4 += 1;
                elif signal4 == "RESP":
                    count_resp4 += 1;
            
                j += 1;
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
          


#%%

find_signals()
#pass dict into a list
x_short_input = x_4_short.values()
#make 5 windows
size = 15000
i = 0
j = 0
z = 0
for k in range(4940):
    print k
    x_short_windowed[k] = x_short_input[j][:,z][i*size:size*(i+1)]
    i += 1;
    if i == 5:
        i = 0
        z += 1
    if z == 4:
        i = 0
        j += 1
        z = 0
        
x_windowed_pepino = np.array(x_short_windowed).reshape(15000,4940)
#normalize the values
scaler = StandardScaler() 
x_short_std = scaler.fit_transform(x_windowed_pepino)
#doing PCA
pca = PCA()
x_short_pca = pca.fit_transform(x_short_std)
eigenvalues_short = pca.explained_variance_
##start with xgboost



