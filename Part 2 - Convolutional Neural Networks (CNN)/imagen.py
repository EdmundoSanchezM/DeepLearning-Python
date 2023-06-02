# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:42:57 2023

@author: josue
"""

from PIL import Image

# Cargar la imagen
imagen = Image.open("dataset/test_set/dogs/dog.4015.jpg")

# Obtener los píxeles de la imagen
pixels = imagen.load()

# Obtener tamaño
anchura, altura = imagen.size

print("Anchura de la imagen:", anchura)
print("Altura de la imagen:", altura)