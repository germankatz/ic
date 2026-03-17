import csv
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP
# Guia 2 - Ejercicio 3


def leer_archivo(nombre):
    datos_entrada = []
    datos_salidas = []
    with open(nombre, mode='r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)

        for fila in lector:
            x1, x2, x3, x4, yd0, yd1, yd2 = map(float, fila)
            datos_entrada.append([x1, x2, x3, x4])  # Ya no agregamos el bias aca, lo hace la capa automáticamente.
            datos_salidas.append([yd0, yd1, yd2])  # Y_deseada

    # Convertir a matrices de NumPy
    X = np.array(datos_entrada)
    Y = np.array(datos_salidas).reshape(-1, 3)  # <- forma (n_patrones, 3)
    return X, Y

def main():
    X, Y = leer_archivo('iris81_trn.csv')
    X_test, Y_test = leer_archivo('iris81_tst.csv')

    tasas = [0.1, 0.01, 0.001]
    colores = ["tab:blue", "tab:orange", "tab:green"]
    resultados = []

    for tasa in tasas:
        print(f"\n=== Entrenando con tasa de aprendizaje = {tasa} ===")
        np.random.seed(0)  # misma inicialización para comparación justa
        red = MLP(estructura=[4,5,5,3], tasa_aprendizaje=tasa, max_epocas=3000)
        history = red.train(X, Y, modo='multiclase')

        # Evaluación en test
        aciertos = 0
        for x, y in zip(X_test, Y_test):
            salida = red.forward(x)
            pos_pred = np.argmax(salida)
            salida_clase = np.ones_like(salida) * -1
            salida_clase[pos_pred] = 1
            if np.array_equal(salida_clase, y):
                aciertos += 1
        tasa_aciertos_test = aciertos / len(X_test)
        print(f"Tasa de aciertos (TEST): {tasa_aciertos_test * 100:.2f}%")

        resultados.append((tasa, history))

    # Gráfica comparativa: ECM y precisión por época para cada tasa
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for (tasa, history), color in zip(resultados, colores):
        ax1.plot(history["ecm"], color=color, label=f"η={tasa}")
        ax2.plot(history["acc"], color=color, label=f"η={tasa}")

    ax1.set_title("ECM vs épocas (comparación de tasas)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Error Cuadrático Medio")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Precisión vs épocas (comparación de tasas)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Tasa de aciertos")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


main()
