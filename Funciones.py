import numpy as np
import matplotlib.pyplot as plt
from time import time 

def generar_red_aleatoria(L):
    """
    Crea una matriz de LxL donde cada elemento (que representa un spin) puede valer 1 o -1 
    
        Args: 
            L (longitud de la red)

        Returns:
            array (L,L)
    """
    if not isinstance(L, int) or L <= 0:
        raise TypeError("L debe ser un entero positivo.")

    red = np.random.choice([-1, 1], size=(L, L))
    return red 



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

def intento_voltear(red, beta):
    """
    Realiza un intento de voltear un espín con el criterio de Metrópolis.

    Args:
        red (numpy.ndarray)
        beta (float)

    Returns:
        aceptado (bool)
    """
    L = red.shape[0]
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    dE = -2.0 * energia_spin(red, i, j)
    if dE <= 0.0:
        red[i, j] *= -1
        return True
    else:
        if np.random.rand() < np.exp(-beta * dE):
            red[i, j] *= -1
            return True
    return False

def paso_MCS(red, beta):
    """
    Ejecuta un Monte Carlo Step (L*L intentos de voltear espines).

    Args:
        red (numpy.ndarray)
        beta (float)

    Returns:
        aceptados (int)
    """
    L = red.shape[0]
    aceptados = 0
    for _ in range(L * L):
        if intento_voltear(red, beta):
            aceptados += 1
    return aceptados

def energia_total(red, J=1.0):
    """
    Calcula la energía total de la red evitando doble conteo.

    Args:
        red (numpy.ndarray)
        J (float)

    Returns:
        energia (float)
    """
    L = red.shape[0]
    E = 0.0
    for i in range(L):
        for j in range(L):
            E -= J * red[i, j] * (red[(i + 1) % L, j] + red[i, (j + 1) % L])
    return E

def magnetizacion_por_spin(red):
    """
    Calcula la magnetización absoluta por espín.

    Args:
        red (numpy.ndarray)

    Returns:
        magnetizacion (float)
    """
    return np.abs(np.sum(red)) / (red.shape[0] * red.shape[1])

def simular_T(L, T, n_equil_MCS=1000, n_muestra_MCS=1000):
    """
    Simula el modelo de Ising para una temperatura dada.

    Args:
        L (int): Tamaño de un lado de la red.
        T (float): Temperatura.
        n_equil_MCS (int): Cantidad de pasos tomados para alcanzar el equilibrio al iniciar.
        n_muestra_MCS (int): Cantidad de pasos entre "mediciones" de las observables.

    Returns:
        lista_energias (numpy.ndarray)
        lista_magnetizaciones (numpy.ndarray)
        red (numpy.ndarray)
    """
    beta = 1.0 / T
    red = generar_red_aleatoria(L)

    for m in range(n_equil_MCS):
        paso_MCS(red, beta)

    energias = []
    magnetizaciones = []

    for m in range(n_muestra_MCS):
        paso_MCS(red, beta)
        E = energia_total(red)
        M = magnetizacion_por_spin(red)
        energias.append(E)
        magnetizaciones.append(M)
    return np.array(energias), np.array(magnetizaciones), red

def barrido_temperaturas(L, temperaturas, n_equil_MCS=1000, n_muestra_MCS=1000):
    """
    Realiza un barrido en temperatura y calcula <M> para cada una.

    Args:
        L (int): Tamaño del lado de la red.
        temperaturas (iterable)
        n_equil_MCS (int)
        n_muestra_MCS (int)

    Returns:
        temps (numpy.ndarray)
        M_prom (numpy.ndarray)
        M_err (numpy.ndarray)
        tiempos (list)
    """
    M_prom = []
    M_err = []
    E_prom = []
    E2_prom = []
    C_cal = []
    tiempos = []
    for T in temperaturas:
        t0 = time()
        energias, magnetizaciones, red_final = simular_T(L, T, n_equil_MCS, n_muestra_MCS)
        tiempo = time() - t0
        tiempos.append(tiempo)
        m_mean = np.mean(magnetizaciones)
        m_std = np.std(magnetizaciones)
        e_mean = np.mean(energias)
        e2_mean = np.mean(energias * energias)
        M_prom.append(m_mean)
        M_err.append(m_std / np.sqrt(len(magnetizaciones)))
        E_prom.append(e_mean)
        E2_prom.append(e2_mean)
        C_cal.append(capacidad_calorifica_por_spin(L, T, e_mean, e2_mean))
        print(f"T={T:.3f}: <M>={m_mean:.4f} ± {M_err[-1]:.4f}  (tiempo {tiempo:.1f}s)")
    return np.array(temperaturas), np.array(M_prom), np.array(M_err), C_cal, tiempos

def capacidad_calorifica_por_spin(L, T, E_prom, E2_prom):
    """
    Calcula la capacidad calorífica a una temperatura dada basada en las fluctuaciones a la energìa.

    Args:
        L (int): Tamaño del lado de la red (cuadrada).
        T (float): Temperatura.
        E_prom: Energía promedio.
        E2_prom: Promedio de las energías cuadradas.

    Returns:
        capacidad calorífica (float)
    """
    return (1/(T*T*L*L))*(E2_prom  - (E_prom * E_prom))
