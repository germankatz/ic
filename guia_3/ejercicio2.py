# Ejercicio 2: Comparar MLP con otros clasificadores usando KFold=5 sobre Digits
# Naive Bayes, LDA, KNN, Arbol de decision, SVM (linear, poly, rbf, sigmoid)

from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import numpy as np

# Cargar dataset
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True)

# Lista de listas para guardar accuracy por clasificador por fold
ACC = [[] for _ in range(9)]

for train_index, test_index in kf.split(X_digits):
    X_train, y_train = X_digits[train_index], y_digits[train_index]
    X_test, y_test = X_digits[test_index], y_digits[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ------------------ MLP ------------------
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
    ACC[0].append(accuracy_score(y_test, clf.predict(X_test)))

    # ------------------ Naive Bayes ------------------
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    ACC[1].append(accuracy_score(y_test, clf.predict(X_test)))

    # ------------------ SVM LINEAR ------------------
    clf = svm.SVC(kernel='linear', C=1, gamma='auto')
    clf.fit(X_train, y_train)
    ACC[2].append(accuracy_score(y_test, clf.predict(X_test)))

    # ------------------ SVM POLINOMIAL ------------------
    clf = svm.SVC(kernel='poly', degree=2, coef0=1, gamma='auto')
    clf.fit(X_train, y_train)
    ACC[3].append(accuracy_score(y_test, clf.predict(X_test)))

    # ------------------ SVM RBF ------------------
    clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
    clf.fit(X_train, y_train)
    ACC[4].append(accuracy_score(y_test, clf.predict(X_test)))

    # ------------------ SVM SIGMOID ------------------
    clf = svm.SVC(kernel='sigmoid', C=1, gamma='auto')
    clf.fit(X_train, y_train)
    ACC[5].append(accuracy_score(y_test, clf.predict(X_test)))

    # ------------------ LDA ------------------
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    ACC[6].append(accuracy_score(y_test, clf.predict(X_test)))

    # ------------------ KNN ------------------
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X_train, y_train)
    ACC[7].append(accuracy_score(y_test, clf.predict(X_test)))

    # ------------------ Arbol de decision ------------------
    clf = DecisionTreeClassifier(random_state=70)
    clf.fit(X_train, y_train)
    ACC[8].append(accuracy_score(y_test, clf.predict(X_test)))

# Resultados
ACC = np.array(ACC)
acc_mean = np.mean(ACC, axis=1)
acc_std = np.std(ACC, axis=1)

classifiers = ["Multicapa", "NaiveBayes", "SVM-linear", "SVM-poly",
               "SVM-RBF", "SVM-sigmoid", "LDA", "kNN", "ArbolDecision"]

table = [["Clasificador", "ACC Mean", "ACC STD"]]
for i in range(len(classifiers)):
    table.append([classifiers[i], '%.4f' % acc_mean[i], '%.4f' % acc_std[i]])

print(tabulate(table, tablefmt="fancy_grid", showindex="always", stralign="center"))
