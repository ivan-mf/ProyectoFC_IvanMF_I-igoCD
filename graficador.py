import numpy as np
import matplotlib.pyplot as plt

T = np.load("Resultados/T.npy")
resultados_M_prom = np.load("Resultados/M_prom.npy")
resultados_M_err = np.load("Resultados/M_err.npy")
resultados_C_cal = np.load("Resultados/C_cal.npy")

print(T)

for M_prom, M_err, C_cal in zip(resultados_M_prom, resultados_M_err, resultados_C_cal):
     plt.scatter(T, M_prom)
     plt.scatter(T, C_cal)

     plt.show()
