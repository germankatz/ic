# Ejercicio 2: Modelo neuronal en PyTorch para clasificacion
# Implementa __init__ y forward() para un MLP (Perceptron Multicapa)
# PyTorch se encarga de la retropropagacion automaticamente

import torch
import torch.nn as nn


class ClasificadorMLP(nn.Module):
    """Perceptron Multicapa para clasificacion.
    Hereda de nn.Module, que es la clase base de todos los modelos en PyTorch.
    Al definir forward(), PyTorch calcula automaticamente los gradientes (backward)."""

    def __init__(self, n_entradas=4, n_ocultas=10, n_salidas=3):
        """Define la arquitectura de la red.
        n_entradas: cantidad de caracteristicas de entrada (4 para Iris)
        n_ocultas: neuronas en la capa oculta
        n_salidas: cantidad de clases (3 para Iris)

        Arquitectura: Entrada(4) -> Oculta(10, tanh) -> Salida(3, tanh)
        Se usa tanh porque las clases estan codificadas en bipolar (-1/1)."""
        super().__init__()
        # Capa oculta: transforma de n_entradas a n_ocultas (incluye bias)
        self.capa_oculta = nn.Linear(n_entradas, n_ocultas)
        self.activacion_oculta = nn.Tanh()  # Tanh: salida en [-1, 1]
        # Capa de salida: transforma de n_ocultas a n_salidas (incluye bias)
        self.capa_salida = nn.Linear(n_ocultas, n_salidas)
        self.activacion_salida = nn.Tanh()  # Tanh en salida para codificacion bipolar

    def forward(self, x):
        """Define el flujo de informacion (propagacion hacia adelante).
        PyTorch usa este metodo para construir el grafo computacional
        y luego calcular los gradientes con backpropagation.

        Flujo: entrada -> capa oculta -> tanh -> capa salida -> tanh -> salida"""
        x = self.capa_oculta(x)       # Transformacion lineal: W1*x + b1
        x = self.activacion_oculta(x)  # Activacion no lineal tanh
        x = self.capa_salida(x)        # Transformacion lineal: W2*x + b2
        x = self.activacion_salida(x)  # Activacion tanh en la salida
        return x


# Prueba del modelo si se ejecuta directamente
if __name__ == "__main__":
    modelo = ClasificadorMLP()
    print(modelo)
    # Prueba con un patron de ejemplo (4 caracteristicas)
    x_ejemplo = torch.tensor([[4.5, 2.3, 1.3, 0.3]])
    salida = modelo(x_ejemplo)
    print(f"Entrada: {x_ejemplo}")
    print(f"Salida:  {salida}")
