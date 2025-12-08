import numpy as np
import matplotlib.pyplot as plt
from time import time 



def generar_red_aleatoria(L):
    """Crea una matriz de LxL donde cada elemento puede valer 1 o -1 
    parámetro: L (longitud de la red)
    devuelve: array (L,L)"""

    return np.random.choice([-1,1], size=(L, L))  



def energia_spin(red, i, j, J=1.0):
    """
    Calcula la energía asociada al espín en la posición (i, j),
    considerando sus vecinos más cercanos y tiene condiciones periódicas.

    Args:
        red (ndarray): Arreglo NumPy 2D con valores +1 y -1.
        i (int): Índice de fila.
        j (int): Índice de columna.
        J (float): Constante de acoplamiento entre espines.

    Returns:
        energia (float): Energía del sitio (i,j) debido a sus vecinos.

    Raises:
        TypeError: si red no es un ndarray, o si sus valores no son ±1,
                   o si i,j están fuera de rango, o si J no es numérico.
    """
     
    if not isinstance(red, np.ndarray):
        raise TypeError("red debe ser un arreglo NumPy.")
    if red[i, j] not in (-1, 1):
        raise TypeError("La red debe contener únicamente valores ±1.")
    if not (0 <= i < red.shape[0] and 0 <= j < red.shape[1]):
        raise TypeError("Índices fuera del rango de la red.")
    if not isinstance(J, (float, int)):
        raise TypeError("J debe ser numérico.")
    L = red.shape[0]

    arriba    = red[(i - 1) % L, j]
    abajo     = red[(i + 1) % L, j]
    izquierda = red[i, (j - 1) % L]
    derecha   = red[i, (j + 1) % L]

    vecindario = arriba + abajo + izquierda + derecha
    energia = -J * red[i, j] * vecindario
    return energia


# Cambio de prueba para subir a GitHub