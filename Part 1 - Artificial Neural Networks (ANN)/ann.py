# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:16:29 2023

@author: josue
"""

# Redes Neuronales Artificales

# Instalar Tensorflow y Keras
# Crear environment conda create -n nombre python=version anaconda
# activate nombre
# conda install spyder
# pip install tensorflow
# pip install keras

# Parte 1 - Pre procesado de datos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Churn_Modelling.csv')


X = dataset.iloc[:, 3:-1].values #Matriz
y = dataset.iloc[:, -1].values  # vector

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_Gender = LabelEncoder()
X[:,2] = labelencoder_X_Gender.fit_transform(X[:,2])

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X), dtype=float)
#Evitar la trampa de las variables dummy.Podemos eliminar cualquiera
X = X[:, 1:]

# Dividir data set entre conjunto de entranamiento y testing
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2, random_state=0)

# Escalado de variables - Obligatorio
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Parte 2 - Construir la RNA
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
"""
11 NODOS DE ENTRADA. CAPA ENTRADA
6 NODOS EN LA SEGUNDA CAPA. CAPA OCULTA
6 NODOS EN LA TERCERA CAPA. CAPA OCULTA
1 NODO CAPA DE SALIDA
"""
# Iniciar la RNA
# 2 formas. 1.- Definir la secuencia de capas
# 2.- Definir el grafo de como se relacionan las capas
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu',
                     input_dim = 11))#Experimentacion

classifier.add(Dropout(p = 0.1))#Empezar con poca probabilidad e ir aumentando
# Añadir la segunda capa oculta. No ocupamos input_dim ya que ya se sabe
classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu')) 

classifier.add(Dropout(p = 0.1))
# Añadir la Capa de salida. Sigmoid da probabilidad
classifier.add(Dense(units=1, kernel_initializer="uniform", activation = 'sigmoid')) 

# Compilar la RNA
# binary_crossentropy: Diferencia y aplicar logaritmo al resultado para 
# transformar las categorias a numeros
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Guardar el modelo
classifier.save("modelo_ann.h5")

# Carga el modelo
classifier = load_model("modelo_ann.h5")

# Verifica la arquitectura del modelo
classifier.summary()

#Parte 3 - Evaluar el modelo y calcular las predicciones finales
# Prediccion de los resultados con el conjunto de Testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5) #1 sale 0 no sale

#Parte 3.1 - Prediccion Individual
"""
Geografia: Francia
Puntaje de crédito: 600
Género masculino
Edad: 40 años de edad
Tenencia: 3 años.
Saldo: $ 60000
Número de productos: 2
¿Este cliente tiene una tarjeta de crédito? Sí
¿Es este cliente un miembro activo? Sí
Salario estimado: $ 50000
Entonces, ¿deberíamos decir adiós a ese cliente?
"""
originalData = [600,'France','Male',40,3,60000,2,1,1,50000]#List
originalData = np.array(originalData).reshape(1, -1)#Array
originalData[:,2] = labelencoder_X_Gender.transform(originalData[:,2])
originalData = np.array(ct.transform(originalData), dtype=float)
originalData = originalData[:, 1:]
new_prediction = classifier.predict(sc_X.transform(originalData))
print(new_prediction)
print(new_prediction>0.5)

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred,labels=[0,1])
print((cm[0][0]+cm[1][1])/cm.sum())

# Parte 4 - Evaluar, mejorar y ajustar RNA
#Evaluar
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    # Iniciar la RNA
    classifier = Sequential()

    # Añadir las capas de entrada y primera capa oculta
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu',
                         input_dim = 11))#Experimentacion
    
    # Añadir la segunda capa oculta. No ocupamos input_dim ya que ya se sabe
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu')) 
    
    # Añadir la Capa de salida. Sigmoid da probabilidad
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation = 'sigmoid')) 
    
    # Compilar la RNA
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train,
                             cv = 10, n_jobs=-1,verbose=1)
#Sesgo. Rendimiento global promedio
accuracies.mean()
#Varianza. a mas baja mejor
accuracies.std()

#Mejorar la RNA
#Regularizacion de DROPOUT para evitar el overfitting

#Ajustar la RNA
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    # Iniciar la RNA
    classifier = Sequential()

    # Añadir las capas de entrada y primera capa oculta
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu',
                         input_dim = 11))#Experimentacion
    
    # Añadir la segunda capa oculta. No ocupamos input_dim ya que ya se sabe
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation = 'relu')) 
    
    # Añadir la Capa de salida. Sigmoid da probabilidad
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation = 'sigmoid')) 
    
    # Compilar la RNA
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])    
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
#Potencia 2: batch_size
parameters = {'batch_size':[25,32],
              'epochs':[100,500],
              'optimizer':['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1) 

grid_search = grid_search.fit(X_train, y_train)

best_accuraacy = grid_search.best_score_

best_parameters = grid_search.best_params_




