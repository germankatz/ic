# Ejercicio 1: Clase Dataset de PyTorch para patrones de entrenamiento y prueba
# Implementa __init__, __len__ y __getitem__
# Carga datos desde CSV con formato: x1,x2,x3,x4,y1,y2,y3

import numpy as np
import torch
from torch.utils.data import Dataset
import csv


class PatronesDataset(Dataset):
    """Clase que hereda de Dataset de PyTorch para manejar los patrones.
    Permite que PyTorch administre los datos con DataLoader (batches, shuffle, etc.)."""

    def __init__(self, archivo_csv):
        """Carga patrones desde un archivo CSV.
        Cada fila tiene el formato: x1, x2, x3, x4, y1, y2, y3
        Las primeras 4 columnas son las caracteristicas (features).
        Las ultimas 3 columnas son la clase codificada en bipolar (-1/1)."""
        self.caracteristicas = []
        self.clases = []

        # Leer el CSV fila por fila
        with open(archivo_csv, mode='r', encoding='utf-8') as f:
            lector = csv.reader(f)
            for fila in lector:
                valores = list(map(float, fila))  # Convertir todos los valores a float
                self.caracteristicas.append(valores[:4])  # Primeras 4: features
                self.clases.append(valores[4:])  # Ultimas 3: clase bipolar

        # Convertir listas a tensores de PyTorch (float32 para compatibilidad con el modelo)
        self.caracteristicas = torch.tensor(np.array(self.caracteristicas), dtype=torch.float32)
        self.clases = torch.tensor(np.array(self.clases), dtype=torch.float32)

    def __len__(self):
        """Retorna la cantidad total de patrones en el dataset."""
        return len(self.caracteristicas)

    def __getitem__(self, idx):
        """Retorna una tupla (caracteristicas, clase) para el patron en la posicion idx.
        Ejemplo: ([4.5, 2.3, 1.3, 0.3], [-1, -1, 1])"""
        return self.caracteristicas[idx], self.clases[idx]


# Prueba del dataset si se ejecuta directamente
if __name__ == "__main__":
    # Cargar datos de entrenamiento
    trn_data = PatronesDataset('guia_5/iris81_trn.csv')
    print(f"Cantidad de patrones de entrenamiento: {len(trn_data)}")
    print(f"Patron 0: {trn_data.__getitem__(0)}")
    print(f"  Caracteristicas: {trn_data[0][0]}")
    print(f"  Clase: {trn_data[0][1]}")

    # Ver cuantas clases distintas hay
    clases_unicas = torch.unique(trn_data.clases, dim=0)
    print(f"\nClases unicas ({len(clases_unicas)}):")
    for i, c in enumerate(clases_unicas):
        print(f"  Clase {i}: {c.tolist()}")

    # Cargar datos de prueba
    tst_data = PatronesDataset('guia_5/iris81_tst.csv')
    print(f"\nCantidad de patrones de prueba: {len(tst_data)}")
    print(f"Patron 0: {tst_data[0]}")
