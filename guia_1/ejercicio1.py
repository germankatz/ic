import csv
import numpy as np
# Guia 1 - Ejercicio 1

# Datos de Entrenamiento
entradas = []
salidas = []

# Abrimos y leemos el archivo, guardamos los valores en los vectores entradas y salidas.
with open('guia_1/OR_trn.csv', mode='r', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
        x1, x2, yd = map(float, fila)
        entradas.append([-1, x1, x2])  # Agregar x0 = -1
        salidas.append(yd) # Y_deseada

# Entrenamiento del Perceptrón
np.random.seed(0)
pesos = np.random.uniform(-0.5, 0.5, 3)  # Vector de 3 valores entre -0.5 y 0.5 ejemplo: [0.35,0.35,0.35]


def funcActivacion(result): # Funcion Signo
    return 1 if result >= 0 else -1

gamma = 0.01 # Tasa de aprendizaje 

max_epocas = 100  # Numero maximo de epocas
for epoca in range(max_epocas):
    errores_epoca = 0
    y = []  # reinicio lista por epoca # Salida y

    for i in range(len(entradas)): # Itera sobre cada patron (fila) de entrenamiento
        potencial = np.dot(entradas[i], pesos)
        y.append(funcActivacion(potencial))

        error = salidas[i] - y[i] # yd - y
        # Actualizamos los pesos
        for j in range(len(pesos)):
            pesos[j] += gamma * error * entradas[i][j] # Si el error = 0 -> no se modifica. 

        if error != 0:
            errores_epoca += 1

    print(f"Época {epoca+1}: errores = {errores_epoca}")

    # Criterio: detener si no hay errores
    if errores_epoca == 0:
        print("Criterio de finalización alcanzado: epoca sin errores.")
        break

# Resultados entrenamiento
print("Pesos modificados:", pesos)

# Calulo de los aciertos en el entrenamiento

y_e = [] # y_entrenados (salida del perceptron con los pesos entrenados)
for i in range(len(entradas)):
    potencial = np.dot(entradas[i], pesos)
    y_e.append(funcActivacion(potencial))

aciertos = 0 # contador de aciertos
for i in range(len(y_e)):
    if y_e[i] == salidas[i]:
        aciertos += 1

print("\nResultados Entrenamiento:")
print("Cantidad de casos entrenamiento:", len(salidas))
print("Casos exitosos entrenamiento:", aciertos)
print("Pesos finales:", pesos)

# Datos de Test

entradastest = []
salidastest = []

with open('guia_1/OR_tst.csv', mode='r', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
        x1, x2, yd = map(float, fila)
        entradastest.append([-1, x1, x2])  # Agregar x0 = -1
        salidastest.append(yd)


# Evaluación con Test (accuracy)

ytest = []
for i in range(len(entradastest)):
    potencial = np.dot(entradastest[i], pesos)
    ytest.append(funcActivacion(potencial))

aciertostest = 0 # contador de aciertos
for i in range(len(ytest)):
    if ytest[i] == salidastest[i]:
        aciertostest += 1

print("\nResultados Test:")
print("Cantidad de casos test:", len(salidastest))
print("Casos exitosos test:", aciertostest)