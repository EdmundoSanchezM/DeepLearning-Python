# Auto Encoders
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 20:02:25 2023

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

# Convertir los datos a tensores de Torch
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# Crear la arquitectura de la Red Neuronal (Auto Encoder) Full Conection
class SAE(nn.Module):#Heredando de class Module
    def __init__(self, ):
        super(SAE, self).__init__()
        #Capa 1. De n peliculas (entrada) a 20 nodos de la capa oculta 1 (salida)
        self.fc1 = nn.Linear(nb_movies, 30)
        #Capa 2. De 20 nodos (entrada) a 10 nodos de la capa oculta 2 (salida)
        self.fc2 = nn.Linear(30, 20)
        #Capa 3. De 10 nodos (entrada) a 20 nodos de la capa oculta 3 (salida)
        self.fc3 = nn.Linear(20, 10)
        #Capa 4. De 20 nodos (entrada) a n peliculas (salida)
        self.fc4 = nn.Linear(30, nb_movies)
        #Funcion activacion para todas las neuronas
        self.activation = nn.Sigmoid()
    def forward(self, x):#x:= vector de datos de 1 usuario
        x = self.activation(self.fc1(x))#Codificado -> activacion. Tam 20
        x = self.activation(self.fc2(x))#Codificado -> activacion. Tam 10
        x = self.activation(self.fc3(x))#Descodificado -> activacion. Tam 20
        x = self.fc4(x)#D0escodificado. Tam original
        return x#prediccion

sae = SAE()
criterion = nn.MSELoss() #Funcion de perdida
#Optimizador(parametros, learningrate(0-1)), decaimiento del peso
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Entrenar el SAE
nb_epoch = 200
for epoch in range(1, nb_epoch+1):
    train_loss = 0
    s = 0.
    for user in training_set:
        val_usr = Variable(user).unsqueeze(0)#Agregando dimension falsa(solo una observacion)
        target = val_usr.clone() #referencia para comparar salida
        if torch.sum(target.data > 0) > 0: #Usuarios con almenos 1 califacion mayor a 0
            output = sae.forward(val_usr)
            target.require_grad = False#No calcular gradiente descendente
            output[target == 0] = 0#Donde la entrada es 0 la salida se conserva
            loss = criterion(output, target)#Calculando perdida
            # la media no es sobre todas las películas, sino sobre las que realmente ha valorado
            mean_corrector = nb_movies/float(torch.sum(target.data > 0)+1e-10) 
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector) ## sum(errors) / n_pelis_valoradas
            s += 1.
            optimizer.step()#corregir los pesos
    print("Epoch: "+str(epoch)+", Loss: "+str(train_loss/s)) #Error de n puntos(estrellas).

# Evaluar el conjunto de test en nuestro SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    val_usr = Variable(training_set[id_user]).unsqueeze(0) #pasado
    target = Variable(test_set[id_user]).unsqueeze(0)#presente
    if torch.sum(target.data > 0) > 0:
        output = sae.forward(val_usr)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        # la media no es sobre todas las películas, sino sobre las que realmente ha valorado
        mean_corrector = nb_movies/float(torch.sum(target.data > 0)+1e-10) 
        test_loss += np.sqrt(loss.data*mean_corrector) ## sum(errors) / n_pelis_valoradas
        s += 1.
print("Test Loss: "+str(test_loss/s)) #Error de n puntos(estrellas).


