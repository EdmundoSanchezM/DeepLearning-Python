# Self Organizing Map
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:52:26 2023

@author: josue
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
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
frauds = np.concatenate((mappings[(1,8)],mappings[(2,8)],mappings[(2,7)]
                         ,mappings[(2,6)],mappings[(3,6)],mappings[(3,5)]),axis=0)
frauds = sc.inverse_transform(frauds) #Potencial tramposo



