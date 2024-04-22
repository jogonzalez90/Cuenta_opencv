# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:13:34 2024

@author: ASUS
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt
 
# Cargamos la imagen
original = cv2.imread("integrado_3.jpg")
plt.imshow(original)
cv2.imshow("original", original)
plt.imshow(original)

# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
plt.imshow(gris) 

# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5,5), 0)
plt.imshow(gauss) 
#plt.imshow(gauss) 

cv2.imshow("suavizado", gauss)
#plt.imshow(suavizado)

# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 10, 625)
#plt.imshow(canny)
cv2.imshow("canny", canny)
plt.imshow(canny)

# Calcular el negativo de la imagen
negativo = 255 - gris
negativo_1 = 255 - canny

# Mostrar la imagen original y el negativo lado a lado
#cv2.imshow('Imagen Original', image)
cv2.imshow('Negativo de la Imagen', negativo)
#cv2.imshow('Negativo de la Imagen', negativo_1)

# Esperar a que se presione una tecla para cerrar las ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()


# Buscamos los contornos
(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mostramos el número de monedas por consola
print("He encontrado {} objetos".format(len(contornos)))


cv2.drawContours(original,contornos,-1,(0,0,255), 2)
cv2.imshow("contornos", original)
#plt.imshow(contornos)
cv2.waitKey(0)






