# Modulo reutilizable: Colonia de Hormigas (ACO)

import numpy as np
import time


class ColoniaHormigas:
    def __init__(self, distancias, n_hormigas=20, iteraciones=100,
                 alpha=1.0, beta=5.0, rho=0.5,
                 feromonas_inicial=1.0, metodo="global"):
        """
        distancias: matriz cuadrada n x n con las distancias entre ciudades (D[i,j] = distancia de i a j)
        n_hormigas: cantidad de hormigas que construyen soluciones por iteración
        iteraciones: número máximo de ciclos
        alpha: exponente que pondera la influencia de las feromonas en la decisión
        beta: exponente que pondera la influencia de la visibilidad (1/distancia) en la decisión
        rho: tasa de evaporación de feromonas (entre 0 y 1)
        feromonas_inicial: valor inicial de feromona en cada arista
        metodo: estrategia de depósito de feromonas:
            - "global":   solo la mejor hormiga global deposita feromona
            - "local":    todas las hormigas depositan proporcionalmente a la calidad de su ruta
            - "uniforme": todas las hormigas depositan una cantidad fija
        """
        self.D = distancias
        self.n = distancias.shape[0]  # cantidad de ciudades
        self.n_hormigas = n_hormigas
        self.iteraciones = iteraciones
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.feromonas_inicial = feromonas_inicial
        self.metodo = metodo

    def calcular_longitud(self, ruta):
        """Calcula la longitud total de una ruta (ciclo completo, vuelve al inicio)."""
        return sum(self.D[ruta[i], ruta[(i + 1) % self.n]] for i in range(self.n))

    def probabilidad_transicion(self, ciudad_actual, ciudades_no_visitadas, feromonas, eta):
        """
        Calcula la probabilidad de ir a cada ciudad no visitada.
        P(ir a j) = (feromona_ij^alpha * eta_ij^beta) / suma
        donde eta_ij = 1/distancia_ij (visibilidad o "conocimiento")
        """
        feromonas_vals = feromonas[ciudad_actual, ciudades_no_visitadas] ** self.alpha
        eta_vals = eta[ciudad_actual, ciudades_no_visitadas] ** self.beta
        probs = feromonas_vals * eta_vals
        return probs / np.sum(probs)

    def construir_ruta(self, feromonas, eta):
        """Una hormiga construye una ruta visitando todas las ciudades."""
        ruta = [np.random.randint(0, self.n)]  # ciudad inicial aleatoria
        while len(ruta) < self.n:
            ciudad_actual = ruta[-1]
            ciudades_no_visitadas = list(set(range(self.n)) - set(ruta))
            p = self.probabilidad_transicion(ciudad_actual, ciudades_no_visitadas, feromonas, eta)
            siguiente = np.random.choice(ciudades_no_visitadas, p=p)
            ruta.append(siguiente)
        return ruta

    def depositar_feromonas(self, feromonas, rutas, longitudes, mejor_ruta, mejor_longitud):
        """Deposita feromonas según el método elegido."""
        if self.metodo == "global":
            # Solo la mejor hormiga global deposita feromona
            for i in range(self.n):
                a, b = mejor_ruta[i], mejor_ruta[(i + 1) % self.n]
                feromonas[a, b] += 1.0 / mejor_longitud
                feromonas[b, a] = feromonas[a, b]  # matriz simétrica

        elif self.metodo == "local":
            # Todas las hormigas depositan proporcionalmente a su calidad
            for k in range(self.n_hormigas):
                for i in range(self.n):
                    a, b = rutas[k][i], rutas[k][(i + 1) % self.n]
                    feromonas[a, b] += 1.0 / longitudes[k]
                    feromonas[b, a] = feromonas[a, b]

        elif self.metodo == "uniforme":
            # Depósito igual por cada hormiga, sin depender de la longitud
            for k in range(self.n_hormigas):
                for i in range(self.n):
                    a, b = rutas[k][i], rutas[k][(i + 1) % self.n]
                    feromonas[a, b] += self.feromonas_inicial
                    feromonas[b, a] = feromonas[a, b]

    def ejecutar(self, semilla=0):
        """
        Ejecuta el algoritmo de colonia de hormigas completo.
        Retorna: (mejor_ruta, mejor_longitud, historia, tiempo, evaluaciones)
            mejor_ruta: lista con el orden de ciudades de la mejor ruta encontrada
            mejor_longitud: longitud total de esa ruta
            historia: lista de la mejor longitud global por iteración (para graficar convergencia)
            tiempo: tiempo de ejecución en segundos
            evaluaciones: cantidad total de rutas construidas
        """
        np.random.seed(semilla)

        # Inicializar feromonas y visibilidad
        feromonas = np.ones((self.n, self.n)) * self.feromonas_inicial
        eta = 1 / (self.D + np.eye(self.n))  # visibilidad: 1/distancia (diagonal con 1 para evitar /0)

        mejor_ruta = None
        mejor_longitud = np.inf
        historia = []
        evaluaciones = 0
        t0 = time.time()

        for it in range(self.iteraciones):
            rutas = []
            longitudes = []

            # Cada hormiga construye una ruta
            for k in range(self.n_hormigas):
                ruta = self.construir_ruta(feromonas, eta)
                longitud = self.calcular_longitud(ruta)
                rutas.append(ruta)
                longitudes.append(longitud)
                evaluaciones += 1

            # Actualizar mejor global
            idx_mejor = np.argmin(longitudes)
            if longitudes[idx_mejor] < mejor_longitud:
                mejor_longitud = longitudes[idx_mejor]
                mejor_ruta = rutas[idx_mejor].copy()

            historia.append(mejor_longitud)

            # Evaporación de feromonas
            feromonas *= (1 - self.rho)

            # Depósito de feromonas según el método
            self.depositar_feromonas(feromonas, rutas, longitudes, mejor_ruta, mejor_longitud)

        tiempo = time.time() - t0
        return mejor_ruta, mejor_longitud, historia, tiempo, evaluaciones
