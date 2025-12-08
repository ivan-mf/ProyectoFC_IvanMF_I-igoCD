from Funciones import *
import matplotlib.pyplot as plt

T = np.linspace(0.5, 5.0, 50)
n_muestra_MCS = 1000

resultados_M_prom = []
resultados_M_err = []
resultados_C_cal = []
resultados_tiempos = []
for L in [10, 20, 30, 40, 50]:
    _, M_prom, M_err, C_cal, tiempos = barrido_temperaturas(L, T, n_muestra_MCS=n_muestra_MCS)

    #Guardamos los resultados mientras van saliendo
    resultados_M_prom.append(M_prom)
    resultados_M_err.append(M_err)
    resultados_C_cal.append(C_cal)
    resultados_tiempos.append(tiempos)

    np.save("Resultados/M_prom", resultados_M_prom)
    np.save("Resultados/M_err", resultados_M_err)
    np.save("Resultados/C_cal", resultados_C_cal)
    np.save("Resultados/tiempos", resultados_tiempos)
