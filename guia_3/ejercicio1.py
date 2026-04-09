# Ejercicio 1: MLP con train_test_split, KFold=5 y KFold=10 sobre Digits
# Analizar desempeno con media y varianza de la tasa de acierto

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

# Cargar dataset de digitos (imagenes 8x8 de numeros escritos a mano del 0 al 9)
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

# Visualizar las primeras 4 imagenes
_, axes = plt.subplots(1, 4, figsize=(10, 4))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap=plt.cm.gray, interpolation="nearest")
    ax.set_title(f"Training: {digits.target[i]}")
    ax.axis("off")
plt.show()

# ==================== 1) Unica particion con train_test_split ====================
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.33, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(20,10), learning_rate_init=0.005, max_iter=300, activation='logistic',
                     early_stopping=True, validation_fraction=0.3, shuffle=True, random_state=0)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

ACC = accuracy_score(y_test, y_pred)
MC = confusion_matrix(y_test, y_pred, labels=digits.target_names)

disp = ConfusionMatrixDisplay(confusion_matrix=MC, display_labels=digits.target_names)
fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, cmap="Blues", values_format='', colorbar=None)
print('ACC (train_test_split):', ACC)
plt.show()

# ==================== 2) y 3) KFold con 5 y 10 particiones ====================
for n_folds in [5, 10]:
    kf = KFold(n_splits=n_folds, shuffle=True)
    acc_list = []

    for train_index, test_index in kf.split(X_digits):
        X_train, y_train = X_digits[train_index], y_digits[train_index]
        X_test, y_test = X_digits[test_index], y_digits[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = MLPClassifier(
            hidden_layer_sizes=(20,10,5),
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
    print(f"\n--- KFold = {n_folds} ---")
    print(f"ACC por fold: {acc_arr}")
    print(f"Media:    {acc_arr.mean():.4f}")
    print(f"Varianza: {acc_arr.var():.4f}")
    print(f"Desvio:   {acc_arr.std():.4f}")
