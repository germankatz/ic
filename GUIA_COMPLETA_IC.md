# Guía Completa de Inteligencia Computacional — UNL FICH

> Informe exhaustivo extraído de las 7 guías de trabajos prácticos, con explicaciones detalladas de cada módulo, parámetros, formatos de archivo y snippets de código comentados.

---

## Tabla de Contenidos

1. [Guía 1 — Perceptrón Simple](#guía-1--perceptrón-simple)
2. [Guía 2 — Perceptrón Multicapa (MLP propio)](#guía-2--perceptrón-multicapa-mlp-propio)
3. [Guía 3 — Métodos de Aprendizaje con Scikit-learn](#guía-3--métodos-de-aprendizaje-con-scikit-learn)
4. [Guía 4 — Aprendizaje No Supervisado (SOM + K-medias)](#guía-4--aprendizaje-no-supervisado-som--k-medias)
5. [Guía 5 — Aprendizaje Profundo con PyTorch](#guía-5--aprendizaje-profundo-con-pytorch)
6. [Guía 6 — Algoritmos Evolutivos (AG)](#guía-6--algoritmos-evolutivos-ag)
7. [Guía 7 — Colonias de Hormigas (ACO) y Enjambre de Partículas (PSO)](#guía-7--colonias-de-hormigas-aco-y-enjambre-de-partículas-pso)
8. [Referencia de Formatos CSV](#referencia-de-formatos-csv)

---

## Guía 1 — Perceptrón Simple

### Qué se pide

Implementar desde cero el entrenamiento y prueba de un perceptrón simple con cantidad variable de entradas. Ejercicio 1: entrenar y probar con OR. Ejercicio 2: visualizar la recta de separación animada para OR y XOR. Ejercicio 3: repetir con datos con 50% y 90% de ruido.

### Formato de los archivos CSV

Cada fila tiene el formato `x1, x2, yd` donde `x1` y `x2` son las entradas y `yd` es la salida deseada (+1 o -1). Ejemplo real de `OR_trn.csv`:

```
0.99941,-1.0007,1
-0.99501,1.0201,1
1.0057,0.98668,1
-0.9801,-1.0187,-1
```

Los puntos se generaron a partir de (1,1), (1,-1), (-1,1), (-1,-1) con desviaciones aleatorias. Las variantes `OR_50_trn.csv` y `OR_90_trn.csv` tienen desviaciones de 50% y 90% respectivamente, lo que hace que los datos sean más difíciles de separar linealmente.

### Cómo leer un archivo CSV

```python
import csv
import numpy as np

entradas = []
salidas = []

with open('guia_1/OR_trn.csv', mode='r', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
        x1, x2, yd = map(float, fila)
        entradas.append([-1, x1, x2])  # -1 es el bias (x0)
        salidas.append(yd)
```

**Punto clave**: el `-1` al inicio de cada entrada es la entrada del bias (`x0 = -1`). El peso `w0` asociado a este bias actúa como el umbral de activación de la neurona. La convención `x0 = -1` (en lugar de `+1`) es una elección del cátedra; con `x0 = -1` el umbral queda como `w0 * (-1)`, de modo que el potencial es `w0*(-1) + w1*x1 + w2*x2`.

### Función de activación (signo)

```python
def funcActivacion(result):
    return 1 if result >= 0 else -1
```

Si la suma ponderada `w · x` es >= 0, la neurona devuelve +1; si no, devuelve -1. Es la función escalón usada en el perceptrón simple clásico.

### Entrenamiento del perceptrón simple

```python
np.random.seed(0)
pesos = np.random.uniform(-0.5, 0.5, 3)  # [w0, w1, w2] inicializados al azar

gamma = 0.01      # Tasa de aprendizaje
max_epocas = 100   # Número máximo de épocas

for epoca in range(max_epocas):
    errores_epoca = 0
    y = []

    for i in range(len(entradas)):
        # 1) Calcular el potencial de activación (producto punto)
        potencial = np.dot(entradas[i], pesos)
        # 2) Aplicar función de activación
        y.append(funcActivacion(potencial))
        # 3) Calcular error
        error = salidas[i] - y[i]   # yd - y
        # 4) Actualizar pesos (regla del perceptrón)
        for j in range(len(pesos)):
            pesos[j] += gamma * error * entradas[i][j]

        if error != 0:
            errores_epoca += 1

    print(f"Época {epoca+1}: errores = {errores_epoca}")

    # Criterio de parada: 0 errores en toda la época
    if errores_epoca == 0:
        print("Convergió: época sin errores.")
        break
```

**Explicación de cada parámetro**:

| Parámetro | Qué controla | Valores típicos | Efecto al variar |
|---|---|---|---|
| `gamma` (tasa de aprendizaje) | Qué tanto se ajustan los pesos en cada paso | 0.001 a 0.1 | Más alto → aprende más rápido pero puede oscilar. Más bajo → convergencia más lenta pero estable. |
| `max_epocas` | Límite de iteraciones sobre todo el dataset | 50 a 1000 | Si los datos no son linealmente separables (ej: XOR), nunca converge y se necesita este tope. |
| `pesos` iniciales | Punto de partida del entrenamiento | Uniformes en [-0.5, 0.5] | Distintas semillas dan distintos tiempos de convergencia. |

**Regla de actualización de pesos**: `Δwj = γ · (yd - y) · xj`. Si la salida es correcta (`error = 0`), los pesos no se modifican. Si hay error, los pesos se mueven proporcionalmente a la entrada y al error.

### Prueba (evaluación) con datos de test

```python
entradastest = []
salidastest = []

with open('guia_1/OR_tst.csv', mode='r', encoding='utf-8') as archivo:
    lector = csv.reader(archivo)
    for fila in lector:
        x1, x2, yd = map(float, fila)
        entradastest.append([-1, x1, x2])
        salidastest.append(yd)

# Evaluar
aciertostest = 0
for i in range(len(entradastest)):
    potencial = np.dot(entradastest[i], pesos)
    ytest = funcActivacion(potencial)
    if ytest == salidastest[i]:
        aciertostest += 1

print(f"Casos test: {len(salidastest)}, Aciertos: {aciertostest}")
```

### Visualización de la recta de separación (Ejercicio 2)

La frontera de decisión del perceptrón simple con 2 entradas es una recta. Se obtiene igualando el potencial a cero: `w0*(-1) + w1*x1 + w2*x2 = 0`, despejando: `x2 = (w0 - w1*x1) / w2`.

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation

pesos_por_epoca = []  # Guardar pesos de cada época durante el entrenamiento

# (después del bucle de entrenamiento)

fig, ax = plt.subplots()
# Graficar puntos de entrenamiento
for i in range(len(entradas)):
    color = "blue" if salidas[i] == 1 else "red"
    marker = "o" if salidas[i] == 1 else "x"
    ax.scatter(entradas[i][1], entradas[i][2], color=color, marker=marker)

x_vals = np.linspace(-2, 2, 100)
linea, = ax.plot([], [], color="black")

def update(frame):
    p = pesos_por_epoca[frame]
    if p[2] != 0:
        y_vals = (p[0] - p[1] * x_vals) / p[2]
        linea.set_data(x_vals, y_vals)
    ax.set_title(f"Época {frame+1}")
    return linea,

ani = animation.FuncAnimation(fig, update, frames=len(pesos_por_epoca),
                               interval=500, repeat=False)
plt.show()
```

**Qué pasa con XOR**: el XOR no es linealmente separable, por lo que el perceptrón simple nunca converge. La recta oscila sin lograr separar las 4 clases. Esto motiva el perceptrón multicapa.

**Qué pasa con ruido alto (50%, 90%)**: con OR_50 y OR_90, los datos se solapan más. A mayor ruido, peor tasa de aciertos en test porque los patrones de entrenamiento pueden estar mal etiquetados en el borde.

---

## Guía 2 — Perceptrón Multicapa (MLP propio)

### Qué se pide

Implementar desde cero el algoritmo de retropropagación para un MLP con cantidad libre de capas y neuronas. Ejercicio 1: XOR. Ejercicio 2: datos concéntricos. Ejercicio 3: Iris (multiclase).

### El módulo `mlp.py` — Arquitectura completa

El MLP se compone de dos clases: `Capa` y `MLP`.

#### Función de activación: sigmoide simétrica

```python
def sigmoide(x):
    return 2 / (1 + np.exp(-x)) - 1   # Rango: [-1, 1]

def sigmoide_derivada(y):
    return 0.5 * (1 + y) * (1 - y)     # Derivada en función de la SALIDA y
```

**Nota**: esta es la sigmoide **simétrica** (también llamada tangente hiperbólica escalada). Su rango es [-1, 1] en lugar de [0, 1]. La derivada se calcula a partir de la salida `y`, no de la entrada `x`, lo cual es más eficiente porque `y` ya se calculó en el forward.

#### Clase `Capa`

```python
class Capa:
    def __init__(self, n_entradas, n_neuronas):
        """
        n_entradas: dimensión de la entrada que recibe esta capa
        n_neuronas: cantidad de neuronas (salidas) de esta capa

        La matriz de pesos tiene tamaño (n_neuronas x n_entradas+1).
        El +1 es por el bias. pesos[i,j] = peso de entrada j hacia neurona i.
        """
        self.pesos = np.random.uniform(-0.5, 0.5, (n_neuronas, n_entradas+1))
        self.salida = None     # Salida de la capa después de forward
        self.delta = None      # Gradiente local (para backpropagation)
        self.entradas = None   # Entradas recibidas (con bias)
```

**Sobre la matriz de pesos**: si una capa tiene 2 entradas y 3 neuronas, `self.pesos` tiene forma `(3, 3)` — 3 neuronas × (2 entradas + 1 bias). La primera columna es el peso del bias.

#### Forward (propagación hacia adelante)

```python
def forward(self, entradas):
    entradas = np.hstack(([-1], entradas))   # Agregar bias x0 = -1
    self.entradas = entradas
    z = np.dot(self.pesos, entradas)          # z_i = Σ w_ij * x_j
    self.salida = sigmoide(z)
    return self.salida
```

**Flujo**: recibe un vector de entradas → le agrega `-1` al inicio como bias → multiplica la matriz de pesos por el vector de entradas → aplica la sigmoide simétrica → devuelve la salida.

#### Cálculo de deltas (backpropagation)

```python
# Delta en la CAPA DE SALIDA:
def calcular_delta_salida(self, yd):
    # δ_i = (yd_i - y_i) * f'(y_i)
    self.delta = (yd - self.salida) * sigmoide_derivada(self.salida)

# Delta en CAPAS OCULTAS:
def calcular_delta_oculta(self, pesos_siguiente, delta_siguiente):
    # δ_i = f'(y_i) * Σ(δ_k * w_ki)   para cada neurona k de la capa siguiente
    self.delta = np.dot(pesos_siguiente[:, 1:].T, delta_siguiente) * sigmoide_derivada(self.salida)
    # pesos_siguiente[:, 1:] descarta los pesos del bias de la capa siguiente
```

**Diferencia clave entre capa de salida y oculta**: en la capa de salida, el error es directo `(yd - y)`. En capas ocultas, el error se "propaga hacia atrás" desde la capa siguiente usando los pesos y deltas de esa capa. Por eso se llama **retropropagación**.

#### Actualización de pesos

```python
def actualizar_pesos(self, tasa_aprendizaje):
    # Δw_ij = tasa * δ_i * x_j
    entradas = self.entradas.reshape(1, -1)   # (1 x n_entradas)
    delta = self.delta.reshape(-1, 1)          # (n_neuronas x 1)
    self.pesos += tasa_aprendizaje * np.dot(delta, entradas)
```

El truco de `reshape` es para poder hacer un producto exterior: `delta (n×1)` × `entradas (1×m)` = `matriz (n×m)` que tiene exactamente la misma forma que `self.pesos`, permitiendo la suma directa.

#### Clase `MLP` — Creación de la red

```python
class MLP:
    def __init__(self, estructura, tasa_aprendizaje=0.1, max_epocas=10000):
        """
        estructura: lista con nº de neuronas por capa.

        Ejemplo: [2, 2, 1]
          → 2 entradas (no es una capa con neuronas)
          → 1 capa oculta con 2 neuronas
          → 1 capa de salida con 1 neurona

        Ejemplo: [4, 5, 4, 2]
          → 4 entradas
          → capa oculta 1: 5 neuronas
          → capa oculta 2: 4 neuronas
          → capa de salida: 2 neuronas
        """
        self.capas = []
        self.tasa = tasa_aprendizaje
        self.max_epocas = max_epocas

        for i in range(1, len(estructura)):
            # Capa i recibe estructura[i-1] entradas y tiene estructura[i] neuronas
            self.capas.append(Capa(estructura[i-1], estructura[i]))
```

**Cómo elegir la estructura**:

| Problema | Estructura sugerida | Por qué |
|---|---|---|
| XOR (2 entradas, 1 salida binaria) | `[2, 2, 1]` | 2 neuronas ocultas bastan para separar el XOR |
| Concéntricos (2 entradas, 1 salida) | `[2, 6, 6, 1]` | Más neuronas porque la frontera es circular |
| Iris (4 entradas, 3 clases) | `[4, 5, 5, 3]` | 4 features, 3 salidas bipolares |

#### Entrenamiento completo (train)

```python
def train(self, X, Y, modo='binario', umbral_error=None):
    """
    X: matriz de entradas (n_patrones x n_entradas)
    Y: matriz de salidas deseadas (n_patrones x n_salidas)
    modo: 'binario' → 1 salida, usa round para contar aciertos
          'multiclase' → varias salidas, usa argmax
    umbral_error: si ECM < umbral_error, detener
    """
    history = {"acc": [], "ecm": []}

    for epoca in range(self.max_epocas):
        error_total = 0
        aciertos = 0

        for x, y in zip(X, Y):
            # === FORWARD ===
            salida = self.forward(x)

            # === BACKWARD ===
            # 1) Delta de la capa de salida
            self.capas[-1].calcular_delta_salida(y)

            # 2) Deltas de capas ocultas (de la penúltima a la primera)
            for i in reversed(range(len(self.capas)-1)):
                self.capas[i].calcular_delta_oculta(
                    self.capas[i+1].pesos,
                    self.capas[i+1].delta
                )

            # 3) Actualizar todos los pesos
            for capa in self.capas:
                capa.actualizar_pesos(self.tasa)

            # Acumular error cuadrático medio
            error_total += np.mean((y - salida)**2)

            # Contar aciertos
            if modo == 'multiclase':
                pos_pred = np.argmax(salida)
                salida_clase = np.ones_like(salida) * -1
                salida_clase[pos_pred] = 1
                if np.array_equal(salida_clase, y):
                    aciertos += 1
            else:
                prediccion = np.round(salida)
                if np.array_equal(prediccion, y):
                    aciertos += 1

        error_total /= len(X)
        tasa_aciertos = aciertos / len(X)
        history["acc"].append(tasa_aciertos)
        history["ecm"].append(error_total)

        if umbral_error is not None and error_total < umbral_error:
            print(f"Convergió en época {epoca+1}")
            break

    return history
```

**Modo binario vs multiclase**: en modo binario (1 sola salida, ej: XOR), la predicción se redondea. En modo multiclase (Iris con 3 salidas bipolares), se toma el `argmax` de la salida y se construye un vector con todo `-1` excepto un `+1` en la posición del máximo.

### Ejercicio 1 — XOR

```python
from mlp import MLP

# Leer datos (2 entradas, 1 salida)
X, Y = leer_archivo('guia_2/XOR_trn.csv')  # X: (n, 2), Y: (n, 1)

# Crear red: 2 entradas → 2 ocultas → 1 salida
red = MLP(estructura=[2, 2, 1], tasa_aprendizaje=0.01, max_epocas=100)
red.train(X, Y)

# Prueba
X_test, Y_test = leer_archivo('guia_2/XOR_tst.csv')
for x, y in zip(X_test, Y_test):
    salida = red.forward(x)
    prediccion = np.round(salida)
    print(f"Entrada: {x}, Esperado: {y}, Predicción: {np.round(salida, 3)}")
```

**Cómo leer el CSV para la Guía 2** (ya sin agregar bias manualmente):

```python
def leer_archivo(nombre):
    datos_entrada = []
    datos_salidas = []
    with open(nombre, mode='r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            x1, x2, yd = map(float, fila)
            datos_entrada.append([x1, x2])      # SIN bias, lo agrega la Capa
            datos_salidas.append(yd)
    X = np.array(datos_entrada)
    Y = np.array(datos_salidas).reshape(-1, 1)   # Forma (n_patrones, 1)
    return X, Y
```

**Diferencia con Guía 1**: en la Guía 1 agregábamos `[-1, x1, x2]` manualmente. En la Guía 2, la clase `Capa` agrega el bias automáticamente dentro de `forward()`.

### Ejercicio 2 — Datos concéntricos

```python
# Estructura más profunda para una frontera no lineal circular
red = MLP(estructura=[2, 6, 6, 1], tasa_aprendizaje=0.01, max_epocas=2000)
history = red.train(X, Y, modo='binario', umbral_error=0.01)
```

**Por qué más neuronas**: los datos concéntricos requieren una frontera de decisión circular (no lineal). Con `[2,6,6,1]` la red tiene capacidad suficiente. Con pocas neuronas (ej: `[2,2,1]`) no lograría separar las clases.

### Ejercicio 3 — Iris (multiclase)

Formato del CSV `iris81_trn.csv`: `x1, x2, x3, x4, yd0, yd1, yd2` donde las clases se codifican en bipolar:

```
5.1,3.3,1.7,0.5,-1,-1,1       ← Setosa:     [-1, -1, 1]
5.6,2.8,4.9,2.0,1,-1,-1       ← Virginica:  [1, -1, -1]
                                  Versicolor: [-1, 1, -1]
```

```python
def leer_archivo_iris(nombre):
    datos_entrada = []
    datos_salidas = []
    with open(nombre, mode='r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo)
        for fila in lector:
            x1, x2, x3, x4, yd0, yd1, yd2 = map(float, fila)
            datos_entrada.append([x1, x2, x3, x4])
            datos_salidas.append([yd0, yd1, yd2])
    X = np.array(datos_entrada)
    Y = np.array(datos_salidas).reshape(-1, 3)  # 3 salidas
    return X, Y

# Comparar distintas tasas de aprendizaje
tasas = [0.1, 0.01, 0.001]
for tasa in tasas:
    np.random.seed(0)  # Misma inicialización para comparación justa
    red = MLP(estructura=[4, 5, 5, 3], tasa_aprendizaje=tasa, max_epocas=3000)
    history = red.train(X, Y, modo='multiclase')
```

**Evaluación multiclase en test**:

```python
aciertos = 0
for x, y in zip(X_test, Y_test):
    salida = red.forward(x)
    pos_pred = np.argmax(salida)         # Índice de la mayor salida
    salida_clase = np.ones_like(salida) * -1
    salida_clase[pos_pred] = 1            # Poner 1 solo en el máximo
    if np.array_equal(salida_clase, y):   # Comparar con la clase deseada
        aciertos += 1
```

**Efecto de la tasa de aprendizaje**: con `η=0.1` converge rápido pero puede oscilar. Con `η=0.001` es muy estable pero lento. Con `η=0.01` es un buen balance. Es recomendable graficar ECM y accuracy vs épocas para cada tasa y compararlas.

---

## Guía 3 — Métodos de Aprendizaje con Scikit-learn

### Qué se pide

Ejercicio 1: MLPClassifier con train_test_split y KFold (5 y 10). Ejercicio 2: comparar MLP con Naive Bayes, LDA, KNN, Árbol de decisión, SVM. Ejercicio 3: Bagging y AdaBoost con dataset Wine.

### Ejercicio 1 — MLPClassifier + Validación cruzada

#### Carga del dataset Digits

```python
from sklearn.datasets import load_digits

digits = load_digits()
X_digits, y_digits = digits.data, digits.target
# X_digits: (1797, 64) → 1797 imágenes de 8x8 píxeles aplanadas
# y_digits: (1797,) → dígitos del 0 al 9
```

#### Opción A: Una única partición con train_test_split

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits,
    test_size=0.33,      # 33% para test
    random_state=42       # Semilla para reproducibilidad
)

clf = MLPClassifier(
    hidden_layer_sizes=(20, 10),  # 2 capas ocultas: 20 y 10 neuronas
    learning_rate_init=0.005,      # Tasa de aprendizaje
    max_iter=300,                  # Épocas máximas
    activation='logistic',         # Sigmoide logística (0 a 1)
    early_stopping=True,           # Detener si validación no mejora
    validation_fraction=0.3,       # 30% del train para validación interna
    shuffle=True,                  # Mezclar datos cada época
    random_state=0
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('ACC:', accuracy_score(y_test, y_pred))
```

**Parámetros de MLPClassifier y su efecto**:

| Parámetro | Qué hace | Efecto al variar |
|---|---|---|
| `hidden_layer_sizes` | Arquitectura de capas ocultas | `(20,10)` = 2 capas. `(100,)` = 1 capa de 100. Más neuronas → más capacidad pero riesgo de overfitting. |
| `learning_rate_init` | Tasa de aprendizaje inicial | Muy alta → diverge. Muy baja → no converge a tiempo. |
| `max_iter` | Épocas máximas | Si se queda corto, el modelo no termina de entrenar. |
| `activation` | Función de activación | `'logistic'` = sigmoide, `'tanh'` = tangente hiperbólica, `'relu'` = ReLU. |
| `early_stopping` | Parada anticipada | `True` → usa validación interna para frenar si deja de mejorar. Evita overfitting. |
| `validation_fraction` | Proporción para validación interna | Solo se usa si `early_stopping=True`. |

#### Opción B: KFold (5 y 10 particiones)

```python
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

for n_folds in [5, 10]:
    kf = KFold(n_splits=n_folds, shuffle=True)
    acc_list = []

    for train_index, test_index in kf.split(X_digits):
        X_train, y_train = X_digits[train_index], y_digits[train_index]
        X_test, y_test = X_digits[test_index], y_digits[test_index]

        # Escalar datos (importante para redes neuronales)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)   # Ajustar y transformar en train
        X_test = scaler.transform(X_test)           # Solo transformar en test

        clf = MLPClassifier(
            hidden_layer_sizes=(20, 10, 5),
            learning_rate_init=0.005,
            max_iter=300,
            activation='logistic',
            early_stopping=True,
            validation_fraction=0.3,
            shuffle=True,
            random_state=0
        )
        clf.fit(X_train, y_train)
        acc_list.append(accuracy_score(y_test, clf.predict(X_test)))

    acc_arr = np.array(acc_list)
    print(f"KFold={n_folds} → Media: {acc_arr.mean():.4f}, Varianza: {acc_arr.var():.4f}")
```

**Por qué StandardScaler**: normaliza cada feature para que tenga media 0 y desvío 1. Las redes neuronales funcionan mejor con datos normalizados. **Siempre** se ajusta el scaler solo con datos de entrenamiento (`fit_transform`) y se aplica al test (`transform`) sin reajustar.

**KFold 5 vs 10**: con 5 folds, cada fold tiene 20% de test. Con 10 folds, cada fold tiene 10% de test. Más folds → estimación más robusta pero más costosa computacionalmente.

### Ejercicio 2 — Comparación de clasificadores

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

# Dentro del loop de KFold:

# Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))

# LDA (Análisis Discriminante Lineal)
clf = LinearDiscriminantAnalysis()

# KNN (K Vecinos Más Cercanos)
clf = KNeighborsClassifier(n_neighbors=7)   # k=7 vecinos

# Árbol de Decisión
clf = DecisionTreeClassifier(random_state=70)

# SVM con distintos kernels
clf = svm.SVC(kernel='linear', C=1, gamma='auto')
clf = svm.SVC(kernel='poly', degree=2, coef0=1, gamma='auto')
clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
clf = svm.SVC(kernel='sigmoid', C=1, gamma='auto')
```

**Parámetros clave de SVM**: `kernel` define la transformación del espacio. `C` controla el margen (C alto → margen estrecho, menos errores en train). `gamma` controla la influencia de cada punto (alto → más local). `degree` es el grado del polinomio para kernel `poly`.

### Ejercicio 3 — Ensambles (Bagging + AdaBoost)

```python
from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

X_wine, y_wine = load_wine(return_X_y=True)

# Bagging con base KNN
bagging = BaggingClassifier(
    KNeighborsClassifier(),     # Clasificador base
    n_estimators=50,             # 50 clasificadores en el ensamble
    max_samples=0.5,             # Cada uno entrena con 50% de los datos
    max_features=0.5,            # Cada uno usa 50% de las features
    bootstrap=True,              # Muestreo con reemplazo
    random_state=42
)
bagging.fit(X_train, y_train)

# AdaBoost (base: árbol de decisión por defecto)
ada = AdaBoostClassifier(n_estimators=50)
ada.fit(X_train, y_train)
```

**Bagging vs AdaBoost**: Bagging entrena clasificadores en paralelo con muestras aleatorias (reduce varianza). AdaBoost entrena secuencialmente, dándole más peso a los ejemplos mal clasificados (reduce bias).

---

## Guía 4 — Aprendizaje No Supervisado (SOM + K-medias)

### Qué se pide

Ejercicio 1: SOM 2D con circulo.csv y te.csv, animado. Repetir con SOM 1D. Ejercicio 2: K-medias e SOM sobre Iris, matrices de contingencia. Ejercicio 3: encontrar k óptimo con métricas de clustering.

### Módulo `som.py` — Self-Organizing Map

#### Crear la matriz de pesos

```python
# SOM 2D: grilla de filas x columnas, cada neurona tiene un vector de 'dimension' componentes
def matriz_vectores_2d(filas, columnas, dimension):
    return np.random.uniform(-0.5, 0.5, (filas, columnas, dimension))
    # Ejemplo: matriz_vectores_2d(5, 5, 2) → forma (5, 5, 2)
    #   → 25 neuronas en una grilla 5x5, cada una con un vector 2D

# SOM 1D: cadena de neuronas
def matriz_vectores_1d(num_neuronas, dimension):
    return np.random.uniform(-0.5, 0.5, (num_neuronas, dimension))
    # Ejemplo: matriz_vectores_1d(25, 2) → forma (25, 2)
```

#### Actualización de pesos (aprendizaje competitivo)

```python
def actualizar_pesos(matriz, x, tasa_de_aprendizaje, radio_vecindad):
    """
    Para cada patrón x:
    1) Encontrar la neurona GANADORA (Best Matching Unit, BMU): la más cercana a x
    2) Actualizar la BMU y sus vecinas (dentro del radio) acercándolas a x
    """
    if matriz.ndim == 2:  # SOM 1D
        distancias = np.linalg.norm(matriz - x, axis=1)
        ganadora = np.argmin(distancias)
        # Actualizar neuronas dentro del radio de vecindad
        for idx in range(max(0, ganadora - radio_vecindad),
                         min(matriz.shape[0], ganadora + radio_vecindad + 1)):
            matriz[idx] += tasa_de_aprendizaje * (x - matriz[idx])

    elif matriz.ndim == 3:  # SOM 2D
        filas, columnas = matriz.shape[:2]
        # Buscar la neurona ganadora (menor distancia a x)
        distancia_min = -1
        ganadora = (0, 0)
        for i in range(filas):
            for j in range(columnas):
                d = np.linalg.norm(matriz[i, j] - x)
                if distancia_min == -1 or d < distancia_min:
                    distancia_min = d
                    ganadora = (i, j)

        # Actualizar vecindad rectangular
        i_g, j_g = ganadora
        for p in range(max(0, i_g - radio_vecindad),
                       min(filas, i_g + radio_vecindad + 1)):
            for q in range(max(0, j_g - radio_vecindad),
                           min(columnas, j_g + radio_vecindad + 1)):
                matriz[p, q] += tasa_de_aprendizaje * (x - matriz[p, q])

    return matriz
```

**Regla de actualización**: `w_new = w_old + η * (x - w_old)`. Cada neurona vecina se mueve "hacia" el patrón de entrada `x`. Cuanto más grande `η`, más se mueve. El radio de vecindad determina cuántas neuronas alrededor de la ganadora también se actualizan.

#### Funciones de decaimiento (tasa y radio)

```python
def vecindad_fun(epoch, radio_max=2):
    """Radio de vecindad que decrece con las épocas."""
    if epoch < 200:
        return radio_max          # Fase de ordenamiento global (radio grande)
    elif epoch < 800:
        t = (epoch - 200) / 600.0
        return max(1, round(radio_max * (1 - t) + 1 * t))  # Decrecimiento lineal
    else:
        return 0                  # Fase de ajuste fino (solo la ganadora)

def tasa_aprendizaje_fun(epoch):
    """Tasa de aprendizaje que decrece con las épocas."""
    if epoch < 200:
        return 0.8                # Alta al inicio (exploración)
    elif epoch < 800:
        t = (epoch - 200) / 600.0
        return 0.8 * (1 - t) + 0.1 * t   # Decrecimiento lineal de 0.8 a 0.1
    else:
        return 0.01               # Baja al final (refinamiento)
```

**Tres fases del SOM**: (1) Fase de ordenamiento (épocas 0-200): radio y tasa altos → el mapa se despliega sobre los datos. (2) Fase de transición (200-800): reducción gradual → refinamiento topológico. (3) Fase de convergencia (800+): solo la ganadora se mueve, con ajuste fino.

#### Entrenamiento del SOM

```python
from som import matriz_vectores_2d, actualizar_pesos, dibujar_som_2d

circulo = cargar_csv('guia_4/circulo.csv')   # Datos: puntos dentro de un círculo
epochs = 1000
matriz = matriz_vectores_2d(5, 5, 2)         # SOM 5x5 para datos 2D

for epoch in range(epochs):
    radio = vecindad_fun(epoch)
    eta = tasa_aprendizaje_fun(epoch)
    for xi in np.random.permutation(circulo):  # Presentar datos en orden aleatorio
        matriz = actualizar_pesos(matriz, xi, eta, radio)
    # Dibujar cada 10 épocas para ver la animación
    if epoch % 10 == 0:
        dibujar_som_2d(ax, matriz, circulo, titulo=f"Época {epoch}")
```

**np.random.permutation**: mezcla el orden de presentación de los datos en cada época. Esto ayuda a que el SOM no tenga sesgos por el orden de los datos.

### Módulo `kmedias.py` — K-means desde cero

```python
def k_means(X, k=3, max_iters=100, tol=1e-4, historial=False):
    """
    X: datos (n_samples x n_features)
    k: número de clusters
    max_iters: iteraciones máximas
    tol: tolerancia — si los centroides se mueven menos que esto, parar
    historial: si True, devuelve la evolución de labels y centroids por iteración

    Retorna: labels (asignación de cluster), centroids, inertia
    """
    n_samples, n_features = X.shape
    # Inicializar centroides: elegir k puntos al azar del dataset
    rng = np.random.default_rng()
    indices = rng.choice(n_samples, size=k, replace=False)
    centroids = X[indices]

    for _ in range(max_iters):
        # 1) Asignar cada punto al centroide más cercano
        distances = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                distances[i, j] = np.linalg.norm(X[i] - centroids[j])
        labels = np.argmin(distances, axis=1)

        # 2) Recalcular centroides como promedio de sus puntos
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            puntos = X[labels == j]
            if len(puntos) > 0:
                new_centroids[j] = puntos.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        # 3) Verificar convergencia
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            break
        centroids = new_centroids

    # Inercia: suma de distancias² de cada punto a su centroide
    inertia = np.sum((X - centroids[labels])**2)
    return labels, centroids, inertia
```

**Parámetros y su efecto**:

| Parámetro | Efecto al variar |
|---|---|
| `k` | Más clusters → más granularidad. k óptimo se busca con métricas (ej: codo, Davies-Bouldin). |
| `tol` | Más bajo → más preciso pero más iteraciones. |
| `max_iters` | Generalmente converge en < 50, pero por seguridad se pone 100. |

### Métricas de clustering

```python
from kmedias import cohesion, separacion, davies_bouldin, fowlkes_mallows, rand_index

# Cohesión (SSW): suma de distancias² al centroide → MENOR es mejor
print(f"Cohesion: {cohesion(X, labels, centroids):.4f}")

# Separación (SSB): distancias² entre centroides y centro global → MAYOR es mejor
print(f"Separacion: {separacion(X, labels, centroids):.4f}")

# Davies-Bouldin: promedio de peor similitud entre clusters → MENOR es mejor
print(f"DB: {davies_bouldin(X, labels, centroids):.4f}")

# Fowlkes-Mallows: comparación con clases reales → 1 = perfecto
print(f"FM: {fowlkes_mallows(y_true, labels):.4f}")

# Rand Index ajustado: comparación con clases reales → 1 = perfecto
print(f"ARI: {rand_index(y_true, labels):.4f}")
```

### Ejercicio 2 — Matrices de contingencia

```python
from sklearn.metrics import confusion_matrix
import pandas as pd

# Comparar tres pares:
# 1) Clases reales vs SOM
contingencia = confusion_matrix(y_true, y_pred_som)
print(pd.DataFrame(contingencia, index=clases, columns=clases))

# 2) Clases reales vs K-means
# 3) K-means vs SOM
```

Para asignar clases a neuronas del SOM: se recorre cada dato, se encuentra la BMU, y se registra la clase real. La clase mayoritaria de cada neurona es la que se le asigna.

```python
from collections import Counter

clases_por_neurona = [[] for _ in range(n_neuronas)]
for dato, clase_real in zip(X, y_true):
    distancias = np.linalg.norm(matriz - dato, axis=1)
    bmu = np.argmin(distancias)
    clases_por_neurona[bmu].append(clase_real)

asignacion_clase = np.full(n_neuronas, -1, dtype=int)
for i in range(n_neuronas):
    if clases_por_neurona[i]:
        asignacion_clase[i] = Counter(clases_por_neurona[i]).most_common(1)[0][0]
```

---

## Guía 5 — Aprendizaje Profundo con PyTorch

### Qué se pide

Ejercicio 1: clase Dataset de PyTorch. Ejercicio 2: modelo nn.Module. Ejercicio 3: entrenamiento con validación y gráficos.

### Ejercicio 1 — Clase Dataset

```python
import torch
from torch.utils.data import Dataset
import csv
import numpy as np

class PatronesDataset(Dataset):
    """Hereda de Dataset para que PyTorch pueda administrar los datos."""

    def __init__(self, archivo_csv):
        self.caracteristicas = []
        self.clases = []

        with open(archivo_csv, mode='r', encoding='utf-8') as f:
            lector = csv.reader(f)
            for fila in lector:
                valores = list(map(float, fila))
                self.caracteristicas.append(valores[:4])   # 4 features
                self.clases.append(valores[4:])             # 3 valores de clase

        # Convertir a tensores float32 (requerido por PyTorch)
        self.caracteristicas = torch.tensor(
            np.array(self.caracteristicas), dtype=torch.float32
        )
        self.clases = torch.tensor(
            np.array(self.clases), dtype=torch.float32
        )

    def __len__(self):
        return len(self.caracteristicas)

    def __getitem__(self, idx):
        """Retorna (features, clase) para el patrón idx."""
        return self.caracteristicas[idx], self.clases[idx]

# Uso:
trn_data = PatronesDataset('guia_5/iris81_trn.csv')
print(len(trn_data))          # Cantidad de patrones
print(trn_data[0])            # Tupla: (tensor([5.1, 3.3, 1.7, 0.5]), tensor([-1., -1., 1.]))
```

**¿Por qué heredar de Dataset?** Permite usar `DataLoader` de PyTorch para iterar en mini-batches, mezclar datos automáticamente, y cargar datos en paralelo.

### Ejercicio 2 — Modelo nn.Module

```python
import torch.nn as nn

class ClasificadorMLP(nn.Module):
    def __init__(self, n_entradas=4, n_ocultas=10, n_salidas=3):
        super().__init__()
        # Capa oculta: n_entradas → n_ocultas (incluye bias automáticamente)
        self.capa_oculta = nn.Linear(n_entradas, n_ocultas)
        self.activacion_oculta = nn.Tanh()
        # Capa de salida: n_ocultas → n_salidas
        self.capa_salida = nn.Linear(n_ocultas, n_salidas)
        self.activacion_salida = nn.Tanh()

    def forward(self, x):
        """PyTorch usa este método para el grafo computacional.
        La retropropagación se calcula automáticamente."""
        x = self.capa_oculta(x)        # W1*x + b1
        x = self.activacion_oculta(x)  # tanh
        x = self.capa_salida(x)         # W2*x + b2
        x = self.activacion_salida(x)   # tanh
        return x
```

**Diferencia con el MLP propio de la Guía 2**: en PyTorch, `nn.Linear` ya maneja los pesos y bias internamente. No necesitás agregar el `-1` manualmente. PyTorch calcula los gradientes automáticamente con `loss.backward()`.

**¿Por qué Tanh y no Sigmoid?** Porque las clases están codificadas en bipolar [-1, 1], y `Tanh` produce salidas en ese rango. Si las clases fueran [0, 1], usaríamos `Sigmoid`.

### Ejercicio 3 — Entrenamiento completo

```python
from torch.utils.data import DataLoader, random_split

# Hiperparámetros
EPOCHS = 300
LR = 0.01
N_OCULTAS = 10
BATCH_SIZE = 16
VAL_RATIO = 0.2

# Cargar datos
trn_data = PatronesDataset('guia_5/iris81_trn.csv')
tst_data = PatronesDataset('guia_5/iris81_tst.csv')

# Separar validación del entrenamiento
n_val = int(len(trn_data) * VAL_RATIO)
n_trn = len(trn_data) - n_val
trn_subset, val_subset = random_split(trn_data, [n_trn, n_val])

# DataLoaders
trn_loader = DataLoader(trn_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=len(val_subset))
tst_loader = DataLoader(tst_data, batch_size=len(tst_data))

# Modelo, loss y optimizador
modelo = ClasificadorMLP(n_entradas=4, n_ocultas=N_OCULTAS, n_salidas=3)
criterio = nn.MSELoss()                              # Error cuadrático medio
optimizador = torch.optim.SGD(modelo.parameters(), lr=LR)  # Descenso por gradiente
```

**Parámetros del entrenamiento PyTorch**:

| Parámetro | Qué hace | Efecto |
|---|---|---|
| `BATCH_SIZE` | Cuántos patrones por paso de gradiente | Más grande → gradiente más estable pero más memoria. Más chico → más ruidoso pero actualiza más seguido. |
| `LR` | Tasa de aprendizaje | Igual que en el MLP propio. |
| `VAL_RATIO` | Proporción para validación | 0.2 = 20% de los datos de train se reservan para monitorear overfitting. |
| `nn.MSELoss()` | Función de pérdida | MSE es buena para salidas continuas en [-1,1]. Alternativa: `CrossEntropyLoss` para clasificación. |

#### Loop de entrenamiento

```python
for epoch in range(EPOCHS):
    modelo.train()             # Modo entrenamiento
    loss_epoch = 0.0
    n_batches = 0

    for x_batch, y_batch in trn_loader:
        optimizador.zero_grad()            # 1) Limpiar gradientes
        salida = modelo(x_batch)           # 2) Forward
        loss = criterio(salida, y_batch)   # 3) Calcular loss
        loss.backward()                    # 4) Backward (retropropagación automática)
        optimizador.step()                 # 5) Actualizar pesos
        loss_epoch += loss.item()
        n_batches += 1

    # Evaluar en validación (sin gradientes)
    modelo.eval()
    with torch.no_grad():
        for x_val, y_val in val_loader:
            loss_val = criterio(modelo(x_val), y_val).item()
```

**Los 5 pasos dentro del loop son siempre los mismos** en PyTorch: `zero_grad()` → forward → loss → `backward()` → `step()`. Esto es el patrón fundamental.

#### Cálculo de accuracy

```python
def calcular_accuracy(modelo, loader):
    modelo.eval()
    correctos = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            salida = modelo(x)
            pred = torch.argmax(salida, dim=1)   # Índice del mayor valor
            real = torch.argmax(y, dim=1)
            correctos += (pred == real).sum().item()
            total += len(y)
    return correctos / total
```

**`torch.no_grad()`**: desactiva el cálculo de gradientes → ahorra memoria y tiempo. Se usa siempre para evaluación.

**`modelo.eval()` vs `modelo.train()`**: cambian el comportamiento de capas como Dropout y BatchNorm. Aunque en este modelo no se usan, es buena práctica siempre cambiar el modo.

---

## Guía 6 — Algoritmos Evolutivos (AG)

### Qué se pide

Ejercicio 1: AG para minimizar funciones f1(x) y f2(x,y), comparar con gradiente descendente. Ejercicio 2: AG para selección de características en datos genómicos de leucemia.

### Módulo `ag.py` — Algoritmo Genético

#### Clase AlgoritmoGenetico

```python
class AlgoritmoGenetico:
    def __init__(self, long_cromosoma, funcion_fitness, poblacion_size=30,
                 generaciones=200, prob_cruza=0.8, prob_mutacion=0.01):
        """
        long_cromosoma: cantidad total de bits por individuo
        funcion_fitness: función que recibe array de 0/1 → retorna número (menor=mejor)
        poblacion_size: individuos en la población (debe ser par para la cruza)
        generaciones: ciclos evolutivos
        prob_cruza: probabilidad de cruzamiento por cada par de padres
        prob_mutacion: probabilidad de flip por cada bit
        """
```

**Parámetros y su efecto**:

| Parámetro | Valores típicos | Si lo subo | Si lo bajo |
|---|---|---|---|
| `poblacion_size` | 30-100 | Más diversidad, más lento | Convergencia prematura |
| `generaciones` | 100-500 | Más tiempo para encontrar el óptimo | Puede cortar antes de converger |
| `prob_cruza` | 0.7-0.9 | Más exploración | Más explotación (copias exactas) |
| `prob_mutacion` | 0.001-0.05 | Más diversidad, puede destruir buenas soluciones | Poca exploración nueva |

#### Flujo del AG

El flujo del AG sigue estos pasos en cada generación:

```
Población inicial (aleatoria)
    ↓
┌──→ Evaluación (fitness de cada individuo)
│       ↓
│   Selección (torneo de tamaño 2)
│       ↓
│   Cruza (un punto, con probabilidad prob_cruza)
│       ↓
│   Mutación (flip de bits con probabilidad prob_mutacion)
│       ↓
│   Nueva generación
│       ↓
└── ¿Terminó? (máx generaciones) → NO → volver arriba
        ↓ SÍ
    Retornar mejor individuo
```

#### Inicialización

```python
def inicializar_poblacion(self):
    """Población aleatoria de 0s y 1s."""
    return np.random.randint(0, 2, (self.poblacion_size, self.long_cromosoma))
    # Ejemplo: 30 individuos de 16 bits → matriz (30, 16) de 0s y 1s
```

#### Selección por torneo

```python
def seleccion_torneo(self, poblacion, valores):
    """Torneo de tamaño 2: elige 2 al azar, gana el de menor fitness."""
    padres = []
    for _ in range(self.poblacion_size):
        i1, i2 = np.random.randint(0, self.poblacion_size, 2)
        if valores[i1] < valores[i2]:   # Menor = mejor
            padres.append(poblacion[i1].copy())
        else:
            padres.append(poblacion[i2].copy())
    return np.array(padres)
```

**Cómo funciona**: de toda la población, se eligen 2 individuos al azar, y el que tiene mejor fitness pasa como padre. Se repite hasta tener tantos padres como individuos hay. Los mejores tienen más chance de ser seleccionados (pero no está garantizado).

#### Cruza de un punto

```python
def cruza(self, padres):
    """Cruza de un punto entre pares consecutivos."""
    hijos = []
    for i in range(0, self.poblacion_size, 2):
        p1 = padres[i]
        p2 = padres[i + 1]
        if np.random.rand() < self.prob_cruza:
            punto = np.random.randint(1, self.long_cromosoma - 1)
            h1 = np.concatenate([p1[:punto], p2[punto:]])
            h2 = np.concatenate([p2[:punto], p1[punto:]])
        else:
            h1, h2 = p1.copy(), p2.copy()
        hijos.append(h1)
        hijos.append(h2)
    return np.array(hijos)
```

**Ejemplo visual de cruza en punto 3**:

```
p1 = [1, 0, 1, | 1, 0, 0]     h1 = [1, 0, 1, | 0, 1, 0]
p2 = [0, 1, 0, | 0, 1, 0]     h2 = [0, 1, 0, | 1, 0, 0]
```

#### Mutación

```python
def mutacion(self, hijos):
    """Por cada bit de cada individuo: si rand < prob_mutacion, flip."""
    for ind in hijos:
        for j in range(self.long_cromosoma):
            if np.random.rand() < self.prob_mutacion:
                ind[j] = 1 - ind[j]   # 0→1 o 1→0
    return hijos
```

#### Decodificación: de binario a valor real

```python
def decode_1var(individuo, dominio, n_bits):
    """Convierte un cromosoma binario a un valor real en [xmin, xmax]."""
    # Paso 1: binario → entero
    valor_entero = 0
    for bit in individuo:
        valor_entero = (valor_entero << 1) | bit
    # Ejemplo: [1,0,1,1] → 1→2→5→11

    # Paso 2: entero → real
    xmin, xmax = dominio
    real = xmin + (valor_entero / (2**n_bits - 1)) * (xmax - xmin)
    return real
    # Ejemplo: 11 de 15 posibles, dominio [-512, 512]:
    # real = -512 + (11/15) * 1024 = 239.47

def decode_2var(individuo, dominio, n_bits):
    """Para 2 variables: la primera mitad de bits es x, la segunda es y."""
    bits_x = individuo[:n_bits]
    bits_y = individuo[n_bits:]
    x = decode_1var(bits_x, dominio, n_bits)
    y = decode_1var(bits_y, dominio, n_bits)
    return x, y
```

**Resolución del cromosoma**: con `n_bits` bits se pueden representar `2^n_bits` valores distintos. Para 16 bits en dominio [-512, 512], la resolución es `1024 / 65535 ≈ 0.0156`. Más bits → más precisión.

### Ejercicio 1 — Minimización de funciones

```python
# Función 1D: f(x) = -x * sin(sqrt(|x|))  en [-512, 512]
def f1(x):
    return -x * np.sin(np.sqrt(np.abs(x)))

# Función fitness: decodifica cromosoma binario → calcula f1
N_BITS_1D = 16
DOMINIO_1D = (-512, 512)

def fitness_1d(individuo):
    x = decode_1var(individuo, DOMINIO_1D, N_BITS_1D)
    return f1(x)

ag = AlgoritmoGenetico(
    long_cromosoma=N_BITS_1D,   # 16 bits por individuo
    funcion_fitness=fitness_1d,
    poblacion_size=50,
    generaciones=400,
    prob_cruza=0.8,
    prob_mutacion=0.01
)
mejor_ind, mejor_f = ag.ejecutar()
mejor_x = decode_1var(mejor_ind, DOMINIO_1D, N_BITS_1D)
print(f"Mejor x = {mejor_x:.5f}, f(x) = {mejor_f:.5f}")
```

**Para la función 2D**: el cromosoma tiene `2 * N_BITS_2D` bits (la primera mitad codifica x, la segunda y).

```python
N_BITS_2D = 12
DOMINIO_2D = (-100, 100)

def fitness_2d(individuo):
    x, y = decode_2var(individuo, DOMINIO_2D, N_BITS_2D)
    return f2(x, y)

ag2 = AlgoritmoGenetico(
    long_cromosoma=2 * N_BITS_2D,   # 24 bits total
    funcion_fitness=fitness_2d,
    poblacion_size=30,
    generaciones=300,
    prob_cruza=0.8,
    prob_mutacion=0.01
)
```

### Ejercicio 2 — Selección de features para leucemia

El dataset tiene 7129 features genómicas. El AG busca el mejor subconjunto.

**Estrategia de pre-filtrado**: como 7129 bits es demasiado, primero se seleccionan las TOP 60 features más discriminativas (mayor diferencia de medias entre clases). El AG busca dentro de esas 60.

```python
# Pre-filtrado: diferencia de medias entre clases
media_clase0 = np.mean(X_train[y_train == 0], axis=0)
media_clase1 = np.mean(X_train[y_train == 1], axis=0)
diferencias = np.abs(media_clase0 - media_clase1)
indices_top = np.argsort(diferencias)[-TOP_N:]  # TOP_N = 60

# Fitness: cada individuo es un vector de 60 bits (1=usar feature, 0=no)
def fitness(individuo):
    mascara = individuo == 1
    if np.sum(mascara) == 0:
        return 1.0  # Peor fitness si no selecciona nada

    X_sel = X_train_norm[:, mascara]

    # Leave-one-out con KNN
    aciertos = 0
    for i in range(len(X_sel)):
        X_loo = np.delete(X_sel, i, axis=0)
        y_loo = np.delete(y_train, i)
        pred = knn_clasificar(X_loo, y_loo, X_sel[i], k=3)
        if pred == y_train[i]:
            aciertos += 1

    accuracy = aciertos / len(X_sel)
    penalizacion = ALPHA * (np.sum(mascara) / TOP_N)  # Premiar parsimonia
    return (1 - accuracy) + penalizacion
```

**La penalización por parsimonia** (`ALPHA = 0.005`): evita que el AG seleccione todas las features. Menos features + buena accuracy = mejor fitness.

---

## Guía 7 — Colonias de Hormigas (ACO) y Enjambre de Partículas (PSO)

### Qué se pide

Ejercicio 1: PSO para minimizar las funciones de la Guía 6, comparar con AG. Ejercicio 2: ACO para el problema del viajante con gr17.csv.

### Módulo `pso.py` — Enjambre de Partículas

```python
class PSO:
    def __init__(self, dimension, funcion_fitness, dominio=(-100, 100),
                 num_particulas=60, iteraciones=100,
                 c1=1.49445, c2=1.49445, vmax_fraction=0.5,
                 max_no_mejora=30):
        """
        dimension: variables del problema (1 para 1D, 2 para 2D)
        funcion_fitness: f(vector) → número (menor=mejor)
        dominio: (xmin, xmax) para todas las dimensiones
        num_particulas: tamaño del enjambre
        c1: coeficiente cognitivo (atracción al mejor personal)
        c2: coeficiente social (atracción al mejor global)
        vmax_fraction: fracción del rango como velocidad máxima
        max_no_mejora: parada anticipada si no mejora
        """
```

**Parámetros del PSO y su efecto**:

| Parámetro | Valores típicos | Efecto al variar |
|---|---|---|
| `num_particulas` | 30-100 | Más → mejor exploración, más lento |
| `c1` (cognitivo) | 1.0-2.0 | Alto → cada partícula explora más cerca de su mejor personal |
| `c2` (social) | 1.0-2.0 | Alto → las partículas convergen más rápido al mejor global |
| `vmax_fraction` | 0.1-0.5 | Alto → movimientos más amplios. Bajo → más preciso. |
| `max_no_mejora` | 20-50 | Parada anticipada para no gastar tiempo si ya convergió |

#### Ecuación de actualización

```python
# Velocidad nueva:
V = V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
V = np.clip(V, -vmax, vmax)   # Limitar velocidad

# Posición nueva:
X = X + V
X = np.clip(X, xmin, xmax)    # Mantener dentro del dominio
```

La velocidad tiene tres componentes: (1) **Inercia**: `V` actual (el movimiento previo). (2) **Componente cognitiva**: `c1 * r1 * (pbest - X)` → atrae hacia el mejor que encontró esta partícula. (3) **Componente social**: `c2 * r2 * (gbest - X)` → atrae hacia el mejor global del enjambre.

`r1` y `r2` son matrices aleatorias en [0,1] que agregan estocasticidad.

#### Uso del PSO

```python
from pso import PSO

# PSO 1D
pso1d = PSO(
    dimension=1,
    funcion_fitness=f1,
    dominio=(-512, 512),
    num_particulas=60,
    iteraciones=100,
    vmax_fraction=0.5
)
x_best, val_best, hist, tiempo, evals, posiciones = pso1d.ejecutar()

# PSO 2D
def f2(xy):
    x, y = xy
    r2 = x**2 + y**2
    return (r2**0.25) * (np.sin(50*(r2**0.1))**2 + 1)

pso2d = PSO(
    dimension=2,
    funcion_fitness=f2,
    dominio=(-100, 100),
    num_particulas=100,
    iteraciones=60,
    vmax_fraction=0.2   # Más bajo para 2D, más precisión
)
xy_best, val_best, hist2D, t2, ev2, pos2D = pso2d.ejecutar()
```

**Lo que retorna `ejecutar()`**: `mejor_posicion` (el punto óptimo), `mejor_valor` (el fitness ahí), `historia` (lista del mejor valor por iteración, para graficar convergencia), `tiempo`, `evaluaciones`, `posiciones` (para graficar las partículas).

### Módulo `aco.py` — Colonia de Hormigas para TSP

#### Problema del Viajante (TSP)

Dada una matriz de distancias `D[i,j]`, encontrar la ruta que visita todas las ciudades exactamente una vez y regresa al inicio, minimizando la distancia total.

El archivo `gr17.csv` es una matriz de distancias 17×17:

```
0,633,257,91,412,150,...
633,0,390,661,227,...
...
```

#### Clase ColoniaHormigas

```python
class ColoniaHormigas:
    def __init__(self, distancias, n_hormigas=20, iteraciones=100,
                 alpha=1.0, beta=5.0, rho=0.5,
                 feromonas_inicial=1.0, metodo="global"):
        """
        distancias: matriz n×n de distancias
        n_hormigas: hormigas por iteración
        alpha: peso de las feromonas en la decisión
        beta: peso de la visibilidad (1/distancia) en la decisión
        rho: tasa de evaporación (0 a 1)
        metodo: "global" | "local" | "uniforme"
        """
```

**Parámetros clave**:

| Parámetro | Efecto |
|---|---|
| `alpha` | Alto → las hormigas siguen más las feromonas (explotación) |
| `beta` | Alto → las hormigas prefieren ciudades cercanas (greedy) |
| `rho` (evaporación) | Alto → olvido rápido, más exploración. Bajo → caminos buenos persisten más. |
| `metodo` | "global" → solo la mejor deposita. "local" → todas depositan proporcional a calidad. "uniforme" → todas depositan igual. |

#### Probabilidad de transición

```python
def probabilidad_transicion(self, ciudad_actual, ciudades_no_visitadas, feromonas, eta):
    """P(ir a j) = (feromona_ij^alpha * eta_ij^beta) / suma_total"""
    feromonas_vals = feromonas[ciudad_actual, ciudades_no_visitadas] ** self.alpha
    eta_vals = eta[ciudad_actual, ciudades_no_visitadas] ** self.beta
    probs = feromonas_vals * eta_vals
    return probs / np.sum(probs)
```

`eta = 1/distancia` → la "visibilidad": ciudades más cercanas son más atractivas. La fórmula combina feromona (experiencia del enjambre) con visibilidad (heurística greedy).

#### Depósito de feromonas según método

```python
# GLOBAL: solo la mejor hormiga global deposita
for i in range(self.n):
    a, b = mejor_ruta[i], mejor_ruta[(i+1) % self.n]
    feromonas[a, b] += 1.0 / mejor_longitud

# LOCAL: todas depositan proporcionalmente
for k in range(self.n_hormigas):
    for i in range(self.n):
        a, b = rutas[k][i], rutas[k][(i+1) % self.n]
        feromonas[a, b] += 1.0 / longitudes[k]

# UNIFORME: todas depositan cantidad fija
feromonas[a, b] += self.feromonas_inicial
```

#### Uso del ACO

```python
import pandas as pd
from aco import ColoniaHormigas

D = pd.read_csv("guia_7/gr17.csv", header=None).values

configuraciones = [
    {"metodo": "global",   "rho": 0.3, "feromonas_inicial": 1.0},
    {"metodo": "global",   "rho": 0.7, "feromonas_inicial": 1.0},
    {"metodo": "local",    "rho": 0.5, "feromonas_inicial": 1.0},
    {"metodo": "uniforme", "rho": 0.5, "feromonas_inicial": 0.1},
]

for conf in configuraciones:
    aco = ColoniaHormigas(
        distancias=D,
        n_hormigas=20,
        iteraciones=100,
        rho=conf["rho"],
        feromonas_inicial=conf["feromonas_inicial"],
        metodo=conf["metodo"]
    )
    ruta, longitud, hist, tiempo, evals = aco.ejecutar()
    print(f"{conf['metodo']} (ρ={conf['rho']}): longitud={longitud:.1f}, t={tiempo:.3f}s")
```

---

## Referencia de Formatos CSV

| Archivo | Formato | Descripción |
|---|---|---|
| `OR_trn.csv` / `OR_tst.csv` | `x1, x2, yd` | OR con 2 entradas y salida ±1 |
| `XOR_trn.csv` / `XOR_tst.csv` | `x1, x2, yd` | XOR con 2 entradas y salida ±1 |
| `OR_50_trn.csv` / `OR_90_trn.csv` | `x1, x2, yd` | OR con 50% y 90% de ruido |
| `concent_trn.csv` / `concent_tst.csv` | `x1, x2, yd` | Clases concéntricas, salida ±1 |
| `iris81_trn.csv` / `iris81_tst.csv` | `x1, x2, x3, x4, yd0, yd1, yd2` | 4 features + 3 salidas bipolares |
| `circulo.csv` | `x1, x2` | Puntos dentro de un círculo (sin labels) |
| `te.csv` | `x1, x2` | Puntos en forma de T (sin labels) |
| `leukemia_train.csv` / `leukemia_test.csv` | `f1,...,f7129, label` | 7129 features + label 0/1 |
| `gr17.csv` | Matriz 17×17 | Distancias entre 17 ciudades (simétrica) |

### Patrón general de lectura

```python
# Para archivos con entradas y salida binaria (Guía 1 y 2)
import csv
datos_entrada, datos_salidas = [], []
with open('archivo.csv', mode='r', encoding='utf-8') as f:
    for fila in csv.reader(f):
        valores = list(map(float, fila))
        datos_entrada.append(valores[:n_features])
        datos_salidas.append(valores[n_features:])

# Para matrices (Guía 7)
import pandas as pd
D = pd.read_csv("gr17.csv", header=None).values

# Para datasets grandes (Guía 6)
datos = np.loadtxt('archivo.csv', delimiter=',')
X = datos[:, :-1]
y = datos[:, -1]
```

---

## Resumen de Configuraciones Usadas por Ejercicio

| Guía | Ejercicio | Módulo/Clase | Configuración clave |
|---|---|---|---|
| G1-E1 | Perceptrón OR | Manual | γ=0.01, 100 épocas, pesos ∈ [-0.5, 0.5] |
| G2-E1 | MLP XOR | `MLP` | `[2,2,1]`, η=0.01, 100 épocas |
| G2-E2 | MLP Concéntricos | `MLP` | `[2,6,6,1]`, η=0.01, 2000 épocas, umbral=0.01 |
| G2-E3 | MLP Iris | `MLP` | `[4,5,5,3]`, η=0.1/0.01/0.001, 3000 épocas, multiclase |
| G3-E1 | MLPClassifier Digits | `sklearn` | `(20,10)` o `(20,10,5)`, η=0.005, 300 iter, logistic |
| G3-E3 | Bagging/AdaBoost Wine | `sklearn` | 50 estimadores, KFold=5 |
| G4-E1 | SOM circulo/te | `som.py` | 5×5, 1000 épocas, radio máx=2 |
| G4-E2 | K-medias Iris | `kmedias.py` | k=3, SOM 1D con 3 neuronas |
| G5-E3 | PyTorch Iris | `nn.Module` | 10 ocultas, LR=0.01, 300 épocas, batch=16 |
| G6-E1 | AG minimización | `ag.py` | 16 bits (1D), 12 bits/var (2D), pop=50, gen=400 |
| G6-E2 | AG feature selection | `ag.py` | 60 bits, pop=30, gen=50, pmut=0.02 |
| G7-E1 | PSO minimización | `pso.py` | 60 partículas (1D), 100 (2D), c1=c2=1.49 |
| G7-E2 | ACO TSP | `aco.py` | 20 hormigas, 100 iter, α=1, β=5 |
