# Ejercicio 3: Bagging y AdaBoost con KFold=5 sobre Wine

from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import numpy as np

# Cargar dataset de vino
X_wine, y_wine = load_wine(return_X_y=True)
print("Datos de x del vino")
print(X_wine)
print("Datos de y del vino")
print(y_wine)

# Configuracion K-Fold
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True)

# ACC[0] -> Bagging, ACC[1] -> AdaBoost
ACC = [[] for _ in range(2)]

for train_index, test_index in kf.split(X_wine):
    X_train, y_train = X_wine[train_index], y_wine[train_index]
    X_test,  y_test  = X_wine[test_index],  y_wine[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # -------------------- Bagging (base: KNN) --------------------
    bagging = BaggingClassifier(
        KNeighborsClassifier(),
        n_estimators=50,
        max_samples=0.5,
        max_features=0.5,
        bootstrap=True,
        bootstrap_features=False,
        random_state=42,
        n_jobs=-1
    )
    bagging.fit(X_train, y_train)
    ACC[0].append(accuracy_score(y_test, bagging.predict(X_test)))

    # -------------------- AdaBoost --------------------
    ada = AdaBoostClassifier(n_estimators=50)
    ada.fit(X_train, y_train)
    ACC[1].append(accuracy_score(y_test, ada.predict(X_test)))

# Resultados
ACC = np.array(ACC)
acc_mean = np.mean(ACC, axis=1)
acc_std  = np.std(ACC, axis=1)

classifiers = ["Bagging", "AdaBoost"]

table = [["Clasificador", "ACC Mean", "ACC STD"]]
for i in range(len(classifiers)):
    table.append([classifiers[i], '%.4f' % acc_mean[i], '%.4f' % acc_std[i]])
print(tabulate(table, tablefmt="fancy_grid", showindex="always", stralign="center"))
