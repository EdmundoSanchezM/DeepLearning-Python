#Maquinas de Boltzmann Restringidas
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 22:51:55 2023

@author: josue
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importar el dataset. #IdPelicula,Nombre(año),Clasificacion
movies = pd.read_csv("ml-1m/movies.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#IdUsr,Sexo,Edad,SectorTrabajo,CP
users  = pd.read_csv("ml-1m/users.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#IdUsr,IdPelicula,Valor,TimeStamp
ratings  = pd.read_csv("ml-1m/ratings.dat", sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparar el conjunto de entrenamiento y elconjunto de testing
#IdUsr,IdPelicula,Valor,TimeStamp
training_set = pd.read_csv("ml-100k/u1.base", sep = "\t", header = None)
training_set = np.array(training_set, dtype = "int")
test_set = pd.read_csv("ml-100k/u1.test", sep = "\t", header = None)
test_set = np.array(test_set, dtype = "int")

# Obtener el número de usuarios y de películas
nb_users = int(max(max(training_set[:, 0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Convertir los datos en un array X[u,i] con usuarios u en fila y películas i en columna
def convert(data):
    new_data = []
    for id_user in range(1, nb_users+1):
        id_movies = data[:, 1][data[:, 0] == id_user]
        id_ratings = data[:, 2][data[:, 0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convertir los datos a tensores de Torch. Matriz de Tensores
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convertir las valoraciones a valores binarios 1 (Me gusta) o 0 (No me gusta)
 
# Crear la arquitectura de la Red Neuronal (Modelo Probabilistico Gráfico)
# Sesgo: probabilidad de que ocurra el nodo oculto conociendo el nodo visible activado y viceversa
# sample: Funcion de activacion, usando el muestreo de Gibbs.Probabilidad de activacion
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv) #Tensor de pesos generados aleatoriamente Tam:nhxnv
        self.a = torch.randn(1, nh)  #Tensor de sego generado aleatoriamente Tam:1xnh
        self.b = torch.randn(1, nv)  #Tensor de sego generado aleatoriamente Tam:1xnv
    #Probilidades y neuronas ocultas que se activaran o no    
    def sample_h(self, x):           #x = mini_batch_size x nv. Observaciones del usr
        wx = torch.mm(x, self.W.t()) #Tam:mini_batch_size x nh. Producto de W x X
        activation = wx + self.a.expand_as(wx) #Tener misma dimension
        p_h_given_v = torch.sigmoid(activation)#Probabilidad de activacion
        return p_h_given_v, torch.bernoulli(p_h_given_v)# , Muestreo bernoulli Tam:mini_batch_size x nh
    #Probilidades y neuronas visibles que se activaran o no    
    def sample_v(self, y):           #y = mini_batch_size x nh. Recomendar o no
        wy = torch.mm(y, self.W)     #mini_batch_size x nv. Producto de W x Y
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)   
    #Divergencia contrastante
    #calificacion original,nodos visibles de k iteraciones, probabilidades de nh, probabilidad de nh despues de
    #k divergencia contrastantes
    def train(self, v0, vk, ph0, phk): 
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

#Numero de nodos visibles, numero de nodos ocultos 
nv = len(training_set[0])
nh = 100
batch_size = 100

rbm = RBM(nv, nh)

# Entrenar la RBM
nb_epoch = 10
for epoch in range(1, nb_epoch+1):
    training_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):#K Pasos de la divergencia contrastante
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0] #Se mantienen los -1
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        training_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print("Epoch: "+str(epoch)+", Loss: "+str(training_loss/s))

# Testear la RBM
testing_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1] #Estado inicial o conocida
    vt = test_set[id_user:id_user+1] #Prediccion
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        testing_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0])) #Comparando solo lsa vistas
        s += 1.
        print("Testing Loss: "+str(testing_loss/s))
