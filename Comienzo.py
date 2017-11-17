#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 03 11:26:39 2017

@author: Fernando Rioja Checa
"""

#Vamos a intentar leer y pintar algunas cosas de la base de datos.

import matplotlib.pyplot as plt
import scipy.io as sp
import sklearn as sk
import numpy as np
import os



def buscar (filename, path):
    
#Le pasamos el nombre de un archivo y nos devuelve la cabecera    
    
    print("Hasta aqui llego")
    
    if str(path) == 'here' or str(path) == 'aqui':
        print('Voy bien')
        path = os.getcwd()
    
    contenido_dir = os.listdir(path)
    header = filename + '.hea'
    for i in contenido_dir:
        if i == str(header):
            header = open(header, 'rw');
            header = header.read();
            print(header)
    
    signals = filename + '.mat'
    for j in contenido_dir:
        if j == str(signals):
            print("Llego hasta pintar");
            senal = sp.loadmat(str(signals))
            print(signals);
            plt.figure(1);
            plt.subplot(311)
            plt.plot(senal['val'][0])            
            plt.subplot(312)
            plt.plot(senal['val'][1])  
            plt.subplot(313)
            plt.plot(senal['val'][2])
            plt.show()

         
buscar(raw_input(), raw_input())        