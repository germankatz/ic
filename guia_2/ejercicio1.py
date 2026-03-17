import csv
import numpy as np
import matplotlib.pyplot as plt
from mlp import MLP
# Guia 2 - Ejercicio 1

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
    # Definir la estructura de la red
    # Abrimos y leemos el archivo, guardamos los valores en los vectores entradas y salidas.
    X, Y = leer_archivo('guia_2/XOR_trn.csv')
    # Crear y entrenar la red
    red = MLP(estructura=[2,2,1], tasa_aprendizaje=0.01, max_epocas=100)
    red.train(X, Y) # Entrenamiento de la red con los datos XOR para obtener los pesos.

    X_test, Y_test = leer_archivo('guia_2/XOR_tst.csv')
    # Prueba final
    print("\nPrueba XOR:")
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

# Obtener los pesos de la capa 1 (dos neuronas) y capa 2 (una neurona)
    pesos_capa1 = red.capas[0].pesos  # (2 neuronas x 3 pesos)
    pesos_capa2 = red.capas[1].pesos  # (1 neurona x 3 pesos)

    xs = np.linspace(-5, 5, 500)

    plt.figure(figsize=(8,6))
    # Graficar puntos
    # Se usan variables para controlar si ya se agregó la etiqueta de cada clase a la leyenda.
    # Antes se usaba "if i == 0" para evitar duplicados, pero eso asume que el punto 0 es siempre Clase 1,
    # lo que no siempre es cierto y podría dejar una clase sin etiqueta en la leyenda.
    label_clase1_usado = False
    label_clase_1_usado = False
    for i in range(len(X)):
        if Y[i] == 1:
            plt.scatter(X[i][0], X[i][1], color="blue", marker="o", label="Clase 1" if not label_clase1_usado else "")
            label_clase1_usado = True
        else:
            plt.scatter(X[i][0], X[i][1], color="red", marker="x", label="Clase -1" if not label_clase_1_usado else "")
            label_clase_1_usado = True

    # Graficar rectas de las neuronas de la capa 1
    colores = ["green", "purple"]
    for idx in range(2): # para cada neurona de la capa 1
        w0, w1, w2 = pesos_capa1[idx]
        ys = (w0 - w1*xs)/w2
        plt.plot(xs, ys, color=colores[idx], label=f"Neurona {chr(97+idx)} (Capa 1)")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title("Fronteras de decisión de las neuronas (XOR)")
    plt.grid(True)
    plt.show()



    salidas_ocultas = []
    for x in X:
        # Propagación hasta la capa oculta
        salida_oculta = red.capas[0].forward(x)
        salidas_ocultas.append(salida_oculta)

    salidas_ocultas = np.array(salidas_ocultas)  # (n_patrones, n_neuronas_ocultas)

    # Graficar las salidas de la anteúltima capa (capa oculta)
    plt.figure(figsize=(8,6))
    # Mismo fix de labels que en la gráfica anterior.
    label_clase1_usado = False
    label_clase_1_usado = False
    for i in range(len(X)):
        if Y[i] == 1:
            plt.scatter(salidas_ocultas[i][0], salidas_ocultas[i][1], color="blue", marker="o", label="Clase 1" if not label_clase1_usado else "")
            label_clase1_usado = True
        else:
            plt.scatter(salidas_ocultas[i][0], salidas_ocultas[i][1], color="red", marker="x", label="Clase -1" if not label_clase_1_usado else "")
            label_clase_1_usado = True

    w0, w1, w2 = pesos_capa2[0]
    # La frontera se calcula en el espacio de salidas ocultas, que están en [-1, 1] por la sigmoide.
    # Se usa ese rango para xs así la recta queda bien ubicada dentro del gráfico.
    xs = np.linspace(-1, 1, 500)
    ys = (w0 - w1*xs)/w2
    plt.plot(xs, ys, color="red", label="Frontera de decisión (Capa 2)")
    plt.title("Salidas de la anteúltima capa (oculta) para cada entrada")
    plt.xlabel("x1")
    plt.ylabel("x2")
    # Las salidas de la capa oculta pasan por la sigmoide simétrica, que devuelve valores en [-1, 1].
    # Con xlim/ylim de [-5, 5] los 4 puntos quedaban apretados en el centro y casi no se veían.
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    plt.legend()
    plt.show()

main()
