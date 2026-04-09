# Modulo reutilizable: Enjambre de Partículas (PSO)

import numpy as np
import time


class PSO:
    def __init__(self, dimension, funcion_fitness, dominio=(-100, 100),
                 num_particulas=60, iteraciones=100,
                 c1=1.49445, c2=1.49445, vmax_fraction=0.5,
                 max_no_mejora=30):
        """
        dimension: cantidad de variables del problema (1 para 1D, 2 para 2D, etc.)
        funcion_fitness: función que recibe un vector de tamaño 'dimension' y retorna un valor numérico
                         (menor = mejor, se minimiza)
        dominio: (xmin, xmax), mismo rango para todas las dimensiones
        num_particulas: cantidad de partículas en el enjambre
        iteraciones: número máximo de ciclos de actualización
        c1: coeficiente de atracción cognitiva (hacia el mejor personal)
        c2: coeficiente de atracción social (hacia el mejor global)
        vmax_fraction: fracción del rango del dominio que limita la velocidad máxima
        max_no_mejora: cantidad de iteraciones sin mejora para parada anticipada
        """
        self.dimension = dimension
        self.funcion_fitness = funcion_fitness
        self.dominio = dominio
        self.num_particulas = num_particulas
        self.iteraciones = iteraciones
        self.c1 = c1
        self.c2 = c2
        self.vmax_fraction = vmax_fraction
        self.max_no_mejora = max_no_mejora

    def inicializar(self):
        """Crea posiciones aleatorias y velocidades en cero."""
        xmin, xmax = self.dominio
        # Posiciones: matriz (num_particulas x dimension)
        X = np.random.uniform(xmin, xmax, size=(self.num_particulas, self.dimension))
        # Velocidades: arrancan en cero
        V = np.zeros_like(X)
        return X, V

    def evaluar(self, X):
        """Evalúa el fitness de cada partícula.
        Si la dimensión es 1, pasa un escalar a la función; si no, pasa el vector."""
        valores = np.zeros(self.num_particulas)
        for i in range(self.num_particulas):
            if self.dimension == 1:
                valores[i] = self.funcion_fitness(X[i, 0])  # escalar
            else:
                valores[i] = self.funcion_fitness(X[i])      # vector
        return valores

    def ejecutar(self, seed=0):
        """
        Ejecuta el PSO completo.
        Retorna: (mejor_posicion, mejor_valor, historia, tiempo, evaluaciones, posiciones)
            mejor_posicion: vector con la mejor posición encontrada
            mejor_valor: valor de fitness en esa posición
            historia: lista del mejor valor global por iteración (para graficar convergencia)
            tiempo: tiempo de ejecución en segundos
            evaluaciones: cantidad total de evaluaciones de la función fitness
            posiciones: lista de arrays con las posiciones de todas las partículas en cada iteración
        """
        np.random.seed(seed)
        xmin, xmax = self.dominio
        vmax = (xmax - xmin) * self.vmax_fraction

        # Inicialización
        X, V = self.inicializar()

        # Mejor personal de cada partícula
        pbest = X.copy()
        pbest_val = self.evaluar(X)

        # Mejor global del enjambre
        g_idx = np.argmin(pbest_val)
        gbest = pbest[g_idx].copy()
        gbest_val = float(pbest_val[g_idx])

        historia = []
        posiciones = []
        no_mejora = 0
        evaluaciones = self.num_particulas
        t0 = time.time()

        for it in range(self.iteraciones):
            # Números aleatorios para la actualización de velocidad
            r1 = np.random.rand(self.num_particulas, self.dimension)
            r2 = np.random.rand(self.num_particulas, self.dimension)

            # Actualizar velocidad:
            #   V = V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
            #   componente cognitiva: c1 * r1 * (pbest - X) → atrae hacia el mejor personal
            #   componente social:    c2 * r2 * (gbest - X) → atrae hacia el mejor global
            V = V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest - X)
            V = np.clip(V, -vmax, vmax)  # limitar velocidad

            # Actualizar posición
            X = X + V
            X = np.clip(X, xmin, xmax)  # mantener dentro del dominio

            # Evaluar fitness de las nuevas posiciones
            vals = self.evaluar(X)
            evaluaciones += self.num_particulas

            # Actualizar mejor personal donde haya mejora
            mejora = vals < pbest_val
            pbest[mejora] = X[mejora]
            pbest_val[mejora] = vals[mejora]

            # Actualizar mejor global
            idx = np.argmin(pbest_val)
            if pbest_val[idx] < gbest_val:
                gbest_val = float(pbest_val[idx])
                gbest = pbest[idx].copy()
                no_mejora = 0
            else:
                no_mejora += 1

            historia.append(gbest_val)
            posiciones.append(X.copy())

            # Parada anticipada si no hay mejora
            if no_mejora >= self.max_no_mejora:
                print(f"Parada anticipada en iteración {it+1} por falta de mejora.")
                break

        tiempo = time.time() - t0

        # Si es 1D, devolver escalar en vez de array de 1 elemento
        if self.dimension == 1:
            gbest = gbest[0]

        return gbest, gbest_val, historia, tiempo, evaluaciones, posiciones
