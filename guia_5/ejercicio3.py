# Ejercicio 3: Entrenamiento y prueba del modelo con Iris
# Usa iris81_trn.csv para entrenamiento (con subconjunto de validacion)
# Usa iris81_tst.csv para prueba
# Grafica curvas de error y muestra accuracy final

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from ejercicio1 import PatronesDataset
from ejercicio2 import ClasificadorMLP

# ==================== Hiperparametros ====================
EPOCHS = 300       # Cantidad de epocas de entrenamiento
LR = 0.01          # Tasa de aprendizaje para SGD
N_OCULTAS = 10     # Neuronas en la capa oculta
BATCH_SIZE = 16    # Tamanio de cada mini-batch
VAL_RATIO = 0.2    # Proporcion de datos de entrenamiento usados para validacion

# ==================== Carga de datos ====================
# Cargar datasets usando la clase del Ejercicio 1
trn_data = PatronesDataset('guia_5/iris81_trn.csv')
tst_data = PatronesDataset('guia_5/iris81_tst.csv')

# Separar una parte del entrenamiento para validacion (evita sobreajuste)
# Validacion sirve para monitorear si el modelo generaliza bien durante el entrenamiento
n_val = int(len(trn_data) * VAL_RATIO)
n_trn = len(trn_data) - n_val
trn_subset, val_subset = random_split(trn_data, [n_trn, n_val])

# DataLoader: permite iterar en mini-batches, con shuffle para entrenamiento
trn_loader = DataLoader(trn_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=len(val_subset))  # Validacion en un solo batch
tst_loader = DataLoader(tst_data, batch_size=len(tst_data))      # Test en un solo batch

print(f"Patrones entrenamiento: {n_trn}")
print(f"Patrones validacion: {n_val}")
print(f"Patrones prueba: {len(tst_data)}")

# ==================== Modelo, loss y optimizador ====================
# Crear el modelo del Ejercicio 2
modelo = ClasificadorMLP(n_entradas=4, n_ocultas=N_OCULTAS, n_salidas=3)
# MSE Loss: error cuadratico medio entre salida del modelo y clase deseada
criterio = nn.MSELoss()
# SGD: descenso por gradiente estocastico para actualizar los pesos
optimizador = torch.optim.SGD(modelo.parameters(), lr=LR)

print(f"\nModelo:\n{modelo}")


# ==================== Funciones auxiliares ====================
def calcular_accuracy(modelo, loader):
    """Calcula el porcentaje de clasificacion correcta (accuracy).
    Compara el argmax de la salida del modelo con el argmax de la clase real.
    argmax convierte la codificacion bipolar a indice de clase (0, 1 o 2)."""
    modelo.eval()  # Modo evaluacion (desactiva dropout, batchnorm, etc.)
    correctos = 0
    total = 0
    with torch.no_grad():  # No calcular gradientes (ahorra memoria y tiempo)
        for x, y in loader:
            salida = modelo(x)
            pred = torch.argmax(salida, dim=1)  # Clase predicha
            real = torch.argmax(y, dim=1)        # Clase real
            correctos += (pred == real).sum().item()
            total += len(y)
    return correctos / total if total > 0 else 0


# ==================== Entrenamiento ====================
print("\n========== Entrenamiento ==========")

# Listas para guardar el historial y graficar despues
hist_loss_trn = []
hist_loss_val = []
hist_acc_trn = []
hist_acc_val = []

for epoch in range(EPOCHS):
    modelo.train()  # Modo entrenamiento
    loss_epoch = 0.0
    n_batches = 0

    # Iterar sobre los mini-batches de entrenamiento
    for x_batch, y_batch in trn_loader:
        optimizador.zero_grad()        # Limpiar gradientes acumulados
        salida = modelo(x_batch)       # Forward: propagar entrada por la red
        loss = criterio(salida, y_batch)  # Calcular error MSE
        loss.backward()                # Backward: calcular gradientes (retropropagacion)
        optimizador.step()             # Actualizar pesos con SGD
        loss_epoch += loss.item()
        n_batches += 1

    loss_trn = loss_epoch / n_batches  # Loss promedio de la epoca

    # Calcular loss de validacion (sin actualizar pesos)
    modelo.eval()
    with torch.no_grad():
        for x_val, y_val in val_loader:
            loss_val = criterio(modelo(x_val), y_val).item()

    # Calcular accuracy en entrenamiento y validacion
    acc_trn = calcular_accuracy(modelo, trn_loader)
    acc_val = calcular_accuracy(modelo, val_loader)

    # Guardar historial
    hist_loss_trn.append(loss_trn)
    hist_loss_val.append(loss_val)
    hist_acc_trn.append(acc_trn)
    hist_acc_val.append(acc_val)

    # Imprimir progreso cada 50 epocas
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"Epoca {epoch:4d} | Loss trn: {loss_trn:.4f} | Loss val: {loss_val:.4f} | "
              f"Acc trn: {acc_trn:.2%} | Acc val: {acc_val:.2%}")

# ==================== Prueba ====================
print("\n========== Prueba ====================")
# Evaluar el modelo final sobre el conjunto de test (datos nunca vistos)
acc_tst = calcular_accuracy(modelo, tst_loader)
print(f"Accuracy en test: {acc_tst:.2%}")

# Obtener predicciones para la matriz de confusion
modelo.eval()
clases = ["Setosa", "Versicolor", "Virginica"]
with torch.no_grad():
    for x_tst, y_tst in tst_loader:
        salida_tst = modelo(x_tst)
        pred_tst = torch.argmax(salida_tst, dim=1)  # Clases predichas
        real_tst = torch.argmax(y_tst, dim=1)        # Clases reales

# Matriz de confusion: filas = clase real, columnas = clase predicha
from sklearn.metrics import confusion_matrix
import pandas as pd

cm = confusion_matrix(real_tst.numpy(), pred_tst.numpy())
df_cm = pd.DataFrame(cm, index=clases, columns=clases)
print(f"\nMatriz de confusion (test):\n{df_cm}")

# ==================== Graficos ====================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Curvas de error (loss) - si validacion sube mientras entrenamiento baja => sobreajuste
axes[0].plot(hist_loss_trn, label='Entrenamiento')
axes[0].plot(hist_loss_val, label='Validacion')
axes[0].set_xlabel('Epoca')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Curvas de error')
axes[0].legend()
axes[0].grid(True)

# Curvas de accuracy - muestra como mejora la clasificacion en cada epoca
axes[1].plot(hist_acc_trn, label='Entrenamiento')
axes[1].plot(hist_acc_val, label='Validacion')
axes[1].set_xlabel('Epoca')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Curvas de accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
