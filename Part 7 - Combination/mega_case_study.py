## Mega Caso de Estudio 
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:43:18 2023

@author: josue
"""

# Parte 1 - Identificar los fraudes potenciales con un SOM

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Escalado de características
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

#Entrenar el SOM
from minisom import MiniSom
#Mapa de x por y. sigma: radio de la vecindad de un nodo
som = MiniSom(x = 10, y = 10, input_len=len(X[0]), sigma = 1.0, learning_rate = 0.5)#Quitando ID cliente
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 200)

#Visualizar los resultados
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)#0-1 mientras mas cerca a 0 mas vecinos hay, caso contrario con 1 hay menos vecinos
colorbar()
makers = ['o','s']#No aprobado, aprobado
colors = ['r','g']#No aprobado, aprobado
#x informacion de cada cliente
for i, x in enumerate(X):
    w = som.winner(x)#Nodo ganador para x
    plot(w[0]+0.5,w[1]+0.5,
         makers[y[i]], markeredgecolor = colors[y[i]], markerfacecolor = 'None',
         markersize = 10, markeredgewidth = 2)
show()

# Encontrar los fraudes
#Mapa de los nodos ganadores, en donde nos muestra el numero y los valores que atrayeron al nodo
mappings = som.win_map(X)
#Colocar las coordenadas de los nodos que su color indique que sea proximo a 1
frauds = np.concatenate((mappings[(4,2)],mappings[(2,3)],mappings[(2,1)]),axis=0)
frauds = sc.inverse_transform(frauds) #Potencial tramposo

# Parte 2 - Trasladar el modelo de Deep Learning de no supervisado a supervisado

# Crear la matriz de características
customers = dataset.iloc[:, 1:-1].values

# Crear la variable dependiente
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
customers = sc_X.fit_transform(customers)

# Parte 2 - Construir la RNA

# Importar Keras y librerías adicionales
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 7, kernel_initializer = "uniform",  activation = "relu", input_dim = 14))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(customers, is_fraud,  batch_size = 1, epochs = 5)

# Predicción de los resultados de fraude
y_pred  = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:,1].argsort()]#Menor a mayor




