import csv
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP
# Guia 2 - Ejercicio 2

np.random.seed(0) # Para que siempre genere los mismos pesos aleatorios.


def leer_archivo(nombre):
    datos_entrada = []
    datos_salidas = []
    with open(nombre, mode='r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)

        for fila in lector:
            x1, x2, yd = map(float, fila)
            datos_entrada.append([x1, x2])  # Ya no agregamos el bias aca, lo hace la capa automáticamente.
            datos_salidas.append(yd) # Y_deseada

    # Convertir a matrices de NumPy
    X = np.array(datos_entrada)
    Y = np.array(datos_salidas).reshape(-1, 1)  # <- forma (n_patrones, 1)
    return X, Y

def main():
    # Abrimos y leemos el archivo, guardamos los valores en los vectores entradas y salidas.
    X, Y = leer_archivo('concent_trn.csv')
    # Crear y entrenar la red
    red = MLP(estructura=[2,6,6,1], tasa_aprendizaje=0.01, max_epocas=2000)
    history = red.train(X, Y, modo='binario', umbral_error=0.01)

    X_test, Y_test = leer_archivo('concent_tst.csv')
    # Prueba final
    print("\nPrueba concent_tst:")
    aciertos = 0
    total = len(X_test)
    for x, y in zip(X_test, Y_test):
        salida = red.forward(x)
        prediccion = np.round(salida)  # Redondea la salida para clasificación binaria
        print(f"Entrada: {x}, Esperado: {y}, Predicción: {np.round(salida, 3)}")
        if np.array_equal(prediccion, y): # si la prediccion es igual a la salida deseada.
            aciertos += 1
    tasa_aciertos = aciertos / total
    print(f"\nTasa de aciertos: {tasa_aciertos * 100:.2f}%")

    # Graficar clasificación predicha por el MLP
    plt.figure(figsize=(8,6))
    for i in range(len(X_test)):
        salida = red.forward(X_test[i])
        prediccion = np.round(salida)
        if prediccion == 1:
            plt.scatter(X_test[i][0], X_test[i][1], color="green", marker="o", label="Predicha 1" if i == 0 else "")
        else:
            plt.scatter(X_test[i][0], X_test[i][1], color="orange", marker="x", label="Predicha -1" if i == 0 else "")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Clasificación predicha por el MLP")
    plt.grid(True)
    plt.show()

    # Graficar evolución de la tasa de aciertos y ECM
    plt.figure(figsize=(8,5))
    plt.plot(history["acc"], label="Precisión del entrenamiento")
    plt.plot(history["ecm"], label="ECM del entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Precisión / ECM")
    plt.title("Evolución de la tasa de aciertos y ECM")
    plt.legend()
    plt.grid(True)
    plt.show()


main()
