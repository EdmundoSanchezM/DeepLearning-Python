# -*- coding: utf-8 -*-
"""
Created on Wed May 17 20:53:17 2023

@author: josue
"""

# Redes Neuronales Convolucionales

# Instalar Tensorflow y Keras
# Crear environment conda create -n nombre python=version anaconda
# activate nombre
# conda install spyder
# pip install tensorflow
# pip install keras

# Parte 1 - Construir el modelo de CNN
import numpy as np
import     
from keras.models import Sequential , load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

#Incializar CNN
classifier = Sequential()

#Paso 1- Convolucion. Potencia a 2, el input_shape depende del tamaño de imagen deben de coincidir
classifier.add(Convolution2D(64, 3, 3, input_shape=(128,128,3),activation="relu"))

#Paso 2- Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

###
classifier.add(Convolution2D(128, 3, 3,activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))
###

#Paso 3- Flattening
classifier.add(Flatten())

#Paso 4- Full Connection
classifier.add(Dense(units=128, kernel_initializer="uniform", activation = 'relu'))#Experimentacion

# Añadir la segunda capa oculta. No ocupamos input_dim ya que ya se sabe
classifier.add(Dense(units=256, kernel_initializer="uniform", activation = 'relu')) 

classifier.add(Dense(units=128, kernel_initializer="uniform", activation = 'relu')) 

# Añadir la Capa de salida. Sigmoid da probabilidad
classifier.add(Dense(units=1, kernel_initializer="uniform", activation = 'sigmoid')) 

# Compilar la CNN
# binary_crossentropy: Diferencia y aplicar logaritmo al resultado para 
# transformar las categorias a numeros
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Parte 2 - Ajustar la CNN a las imagenes para entrenar
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(128, 128),
                                                    batch_size=16,
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(128, 128),
                                                batch_size=16,
                                                class_mode='binary')

classifier.fit(training_dataset,
                        steps_per_epoch=8000/16,
                        epochs=50,
                        validation_data=testing_dataset,
                        validation_steps=2000/16)

# Guardar el modelo
classifier.save("modelo_cnn.h5")

# Carga el modelo
classifier = load_model("modelo_cnn.h5")

# Verifica la arquitectura del modelo
classifier.summary()

#Parte 3 - Evaluar el modelo y calcular las predicciones finales
testing_image = test_datagen.flow_from_directory('dataset/test_image',
                                                target_size=(128, 128),
                                                batch_size=20,
                                                class_mode='binary')

# Prediccion de los resultados con el conjunto de Testing Image
y_pred = classifier.predict(testing_image)
y_pred = (y_pred>0.5)

# Obtención de las etiquetas verdaderas del conjunto de prueba
y_true_pred = testing_image.classes

# Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true_pred, y_pred,labels=[0, 1])

# Obtener los nombres de las clases del generador de datos
class_names = list(training_dataset.class_indices.keys())

# Obtener los nombres de las imágenes
image_names = testing_image.filenames

# Imprimir el nombre de la primera imagen y su correspondiente predicción
print("Nombre de la imagen:", image_names[13])
print("Predicción:", y_pred[13])

#Parte 3.5 - Predicciones individuales
import numpy as np
import keras.utils as image
test_image = image.load_img('dataset/test_image/dogs/IMG_20230516_154026820.jpg',
                              target_size=(128,128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = (classifier.predict(test_image)>0.5)
training_dataset.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

# Parte 4 - Evaluar, mejorar y ajustar RNA
#Evaluar
accuracy = classifier.evaluate(training_dataset)
print('Precisión del clasificador:', accuracy[1])


































