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
    
#Le pasamos el nombre de un archivo y la ruta del directorio
#y nos devuelve la cabecera del fichero.
    
    four_signals = False;    
    if str(path) == 'here' or str(path) == 'aqui': #Si ponemos aqui/here coge
    #el directorio actual
        path = os.getcwd()
    
    contenido_dir = os.listdir(path)
    header = filename + '.hea'
    for i in contenido_dir:
        if i == str(header):
            header = open(header, 'rw');
            lineas = header.readlines();
            linea1 = lineas[1].split();
            linea2 = lineas[2].split();
            linea3 = lineas[3].split();
            
            Signal1 = linea1[-1];
            Signal2 = linea2[-1];
            Signal3 = linea3[-1];

            if ' 4 ' in lineas[0]: #Miramos si tenemos 4 señales o 3
                four_signals = True;
                linea4 = lineas[4].split();
                Signal4 = linea4[-1];

                    
    signals = filename + '.mat'
    for j in contenido_dir:
        if j == str(signals):
            print("Llego hasta pintar");
            senal = sp.loadmat(str(signals))
            print(signals);
            plt.figure(1);
            
            if four_signals == False: #Tenemos 3 señales del paciente
                plt.subplot(311)
                plt.plot(senal['val'][0])            
                plt.subplot(312)
                plt.plot(senal['val'][1])  
                plt.subplot(313)
                plt.plot(senal['val'][2])
                plt.show()
            else:                   #Tenemos 4 señales en este caso
                plt.subplot(411)
                plt.plot(senal['val'][0])            
                plt.subplot(412)
                plt.plot(senal['val'][1])  
                plt.subplot(413)
                plt.plot(senal['val'][2])
                plt.subplot(414)
                plt.plot(senal['val'][3])
                plt.show()
            
    a = min(senal['val'][2])
    print(a)

         
buscar(raw_input(), raw_input())        