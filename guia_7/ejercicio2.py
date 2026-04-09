import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aco import ColoniaHormigas

# Ejercicio 2 - Guía 7
# Colonia de hormigas para Agente viajero

# Cargar matriz de distancias (archivo gr17.csv)
D = pd.read_csv("guia_7/gr17.csv", header=None).values
n = D.shape[0]  # numero de ciudades


# Experimentos con distintas configuraciones

configuraciones = [
    {"metodo": "global", "rho": 0.3, "feromonas_inicial": 1.0},
    {"metodo": "global", "rho": 0.7, "feromonas_inicial": 1.0},
    {"metodo": "local",  "rho": 0.5, "feromonas_inicial": 1.0},
    {"metodo": "uniforme","rho": 0.5, "feromonas_inicial": 0.1},
]

resultados = []

for conf in configuraciones:
    aco = ColoniaHormigas(
        distancias=D,
        n_hormigas=20,
        iteraciones=100,
        rho=conf["rho"],
        feromonas_inicial=conf["feromonas_inicial"],
        metodo=conf["metodo"]
    )
    ruta, longitud, hist, t, evals = aco.ejecutar()

    resultados.append({
        "Método": conf["metodo"],
        "ρ": conf["rho"],
        "Feromona inicial": conf["feromonas_inicial"],
        "Longitud mínima": round(longitud, 3),
        "Tiempo (s)": round(t, 3),
        "Evaluaciones": evals
    })
    plt.plot(hist, label=f"{conf['metodo']} (ρ={conf['rho']})")


# Resultados y tabla comparativa
plt.xlabel("Iteración")
plt.ylabel("Mejor longitud encontrada")
plt.title("Convergencia de la Colonia de Hormigas")
plt.legend()
plt.grid(True)
plt.show()

tabla_resultados = pd.DataFrame(resultados)
print("\n TABLA COMPARATIVA DE RESULTADOS ")
print(tabla_resultados)
