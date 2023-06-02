# Redes Neuronales Recurrentes (RNR)
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:06:38 2023

@author: josue
"""

# Parte 1 - Preprocesado de los datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
dataset_train = dataset_train.replace(',','', regex=True)
training_set = dataset_train.iloc[:, 1:6].values

#Escalado de caracteristicas. Normalizacion
from sklearn.preprocessing import MinMaxScaler
SC = MinMaxScaler(feature_range=(0,1))
training_set_scaled = SC.fit_transform(training_set)

# Crear una estructura de datos con 60 timesteps y 1 salida
# Hasta 60 pasos hacia atras y predecir 1 paso mas 
X_train_Open = []
X_train_High = []
X_train_Low = []
X_train_Close = []
X_train_Volume = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train_Open.append(training_set_scaled[i-60:i,0])
    X_train_High.append(training_set_scaled[i-60:i,1])
    X_train_Low.append(training_set_scaled[i-60:i,2])
    X_train_Close.append(training_set_scaled[i-60:i,3])
    X_train_Volume.append(training_set_scaled[i-60:i,4])
    y_train.append(training_set_scaled[i,0])
X_train_Open = np.array(X_train_Open)
X_train_High = np.array(X_train_High)
X_train_Low = np.array(X_train_Low)
X_train_Close = np.array(X_train_Close)
X_train_Volume = np.array(X_train_Volume)
y_train = np.array(y_train)

# Redimension de los datos.Añadir mas indicadores
X_train_Open = np.reshape(X_train_Open, (X_train_Open.shape[0],X_train_Open.shape[1],1))
X_train_High = np.reshape(X_train_High, (X_train_High.shape[0],X_train_High.shape[1],1))
X_train_Low = np.reshape(X_train_Low, (X_train_Low.shape[0],X_train_Low.shape[1],1))
X_train_Close = np.reshape(X_train_Close, (X_train_Close.shape[0],X_train_Close.shape[1],1))
X_train_Volume = np.reshape(X_train_Volume, (X_train_Volume.shape[0],X_train_Volume.shape[1],1))

X_train = np.append(X_train_Open, (X_train_High), axis = 2)
X_train = np.append(X_train, (X_train_Low), axis = 2)
X_train = np.append(X_train, (X_train_Close), axis = 2)
X_train = np.append(X_train, (X_train_Volume), axis = 2)
# Parte 2 - Construcción de la RNR
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout

#Inicializador del modelo
regressor = Sequential()

#Añadir la primera capa de LSTM y la regularizacion por Dropout
#Debe de ser un numero grande por la dimensionalidad
#Conectar al LSTM posterior
regressor.add(LSTM(units=60, return_sequences = True, input_shape=(X_train.shape[1],5)))
regressor.add(Dropout(rate=0.2))

#Añadir la primera capa de LSTM y la regularizacion por Dropout
regressor.add(LSTM(units=60, return_sequences = True))
regressor.add(Dropout(rate=0.2))

#Añadir la primera capa de LSTM y la regularizacion por Dropout
regressor.add(LSTM(units=60, return_sequences = True))
regressor.add(Dropout(rate=0.2))

#Añadir la primera capa de LSTM y la regularizacion por Dropout
regressor.add(LSTM(units=60, return_sequences = False))
regressor.add(Dropout(rate=0.2))

#Añadir la capa de salida
regressor.add(Dense(units=1))

# Compilar la RNR
regressor.compile(optimizer = "adam", loss = "mean_squared_error")

# Ajustamos la RNR al Conjunto de Entrenamiento
regressor.fit(X_train, y_train, epochs = 200, batch_size=32)
               
# Guardar el modelo
regressor.save("modelo_rnn_all.h5")

# Carga el modelo
regressor = load_model("modelo_rnn_all.h5")

# Verifica la arquitectura del modelo
regressor.summary()

# Parte 3 - Ajustar las predicciones y visualizar los resultados
#Obtener el valor real
dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
dataset_test = dataset_test.replace(',','', regex=True)
real_stock_price = dataset_test.iloc[:, 1:2].values

#Predecir las acciones de Enero de 2017 con la RNR
dataset_total = pd.concat((dataset_train[['Open','High','Low','Close','Volume']],
                           dataset_test[['Open','High','Low','Close','Volume']]),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
#inputs = inputs.reshape(-1,1)
inputs = SC.transform(inputs)
# Crear una estructura de datos con 60 timesteps y 1 salida
# Hasta 60 pasos hacia atras y predecir 1 paso mas 
X_test_Open = []
X_test_High = []
X_test_Low = []
X_test_Close = []
X_test_Volume = []
for i in range(60, len(inputs)):
    X_test_Open.append(inputs[i-60:i,0])
    X_test_High.append(inputs[i-60:i,1])
    X_test_Low.append(inputs[i-60:i,2])
    X_test_Close.append(inputs[i-60:i,3])
    X_test_Volume.append(inputs[i-60:i,4])
X_test_Open = np.array(X_test_Open)
X_test_High = np.array(X_test_High)
X_test_Low = np.array(X_test_Low)
X_test_Close = np.array(X_test_Close)
X_test_Volume = np.array(X_test_Volume)

# Redimension de los datos.Añadir mas indicadores
X_test_Open = np.reshape(X_test_Open, (X_test_Open.shape[0],X_test_Open.shape[1],1))
X_test_High = np.reshape(X_test_High, (X_test_High.shape[0],X_test_High.shape[1],1))
X_test_Low = np.reshape(X_test_Low, (X_test_Low.shape[0],X_test_Low.shape[1],1))
X_test_Close = np.reshape(X_test_Close, (X_test_Close.shape[0],X_test_Close.shape[1],1))
X_test_Volume = np.reshape(X_test_Volume, (X_test_Volume.shape[0],X_test_Volume.shape[1],1))

X_test = np.append(X_test_Open, (X_test_High), axis = 2)
X_test = np.append(X_test, (X_test_Low), axis = 2)
X_test = np.append(X_test, (X_test_Close), axis = 2)
X_test = np.append(X_test, (X_test_Volume), axis = 2)

predicted_stock_price = regressor.predict(X_test)

#model.predict nos devuelve un array de tamaño (20, 1) el cual no podemos 
#pasarlo al scaler creado ya que espera un array de (20, 5), por lo que le 
#agregamos 4 columnas más con valor 0:

predicted_stock_price = np.append(predicted_stock_price, ([[0]]*len(dataset_test)), axis = 1)
predicted_stock_price = np.append(predicted_stock_price, ([[0]]*len(dataset_test)), axis = 1)
predicted_stock_price = np.append(predicted_stock_price, ([[0]]*len(dataset_test)), axis = 1)
predicted_stock_price = np.append(predicted_stock_price, ([[0]]*len(dataset_test)), axis = 1)
predicted_stock_price = SC.inverse_transform(predicted_stock_price)
#Nota ver paso 7 RNN mas de una variable preguntas
# Visualizar los Resultados
plt.plot(real_stock_price, color = 'red', label = 'Precio Real de la Accion de Google')
plt.plot(predicted_stock_price[:, 0], color = 'blue', label = 'Precio Predicho de la Accion de Google')
plt.title("Prediccion con una RNR del valor de las acciones de Google")
plt.xlabel("Fecha")
plt.ylabel("Precio de la accion de Google")
plt.legend()
plt.show()