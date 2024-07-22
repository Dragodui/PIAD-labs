import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import time

# Задание 1

# Генерация случайного набора данных с 200 образцами, 2 признаками, 2 информативными, 0 избыточными,
# 2 кластерами на класс и 2 классами
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=2, n_classes=2, random_state=42)

# Инициализация словаря классификаторов
classifiers = {
    "GaussianNB": GaussianNB(),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(probability=True),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

# Визуализация сгенерированного набора данных
plt.figure()
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.legend()
plt.title('Пример визуализации случайных объектов в 2 классах')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.show()

# Определение списка названий метрик и соответствующих функций
metric_names = ['accuracy_score', 'recall_score', 'precision_score', 'f1_score', 'roc_auc']
metrics_funcs = [metrics.accuracy_score, metrics.recall_score, metrics.precision_score, metrics.f1_score,
                 metrics.roc_auc_score]

# Инициализация словаря для хранения результатов для каждого классификатора
results = {name: [] for name in classifiers.keys()}

# Запуск 100 итераций разделения на тренировочный и тестовый наборы, обучения, предсказания и оценки
for i in range(100):
    # Разделение данных на тренировочный и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

    # Обучение и оценка каждого классификатора
    for clf_name, clf in classifiers.items():
        start_train = time.time()
        clf.fit(X_train, y_train)  # Обучение классификатора
        train_time = time.time() - start_train

        start_test = time.time()
        y_pred = clf.predict(X_test)  # Предсказание для тестового набора
        test_time = time.time() - start_test

        # Вычисление метрик оценки
        metrics_result = [func(y_test, y_pred) for func in metrics_funcs]

        # Вычисление ROC AUC
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            y_proba = clf.decision_function(X_test)
        roc_auc = metrics.roc_auc_score(y_test, y_proba)

        # Добавление результатов в словарь
        results[clf_name].append([train_time, test_time] + metrics_result[:4] + [roc_auc])

# Вычисление средних результатов для каждого классификатора
avg_results = {name: np.mean(scores, axis=0) for name, scores in results.items()}
df_results = pd.DataFrame.from_dict(avg_results, orient='index', columns=['train_time', 'test_time'] + metric_names)

# Построение графика результатов
df_results.plot(kind='bar')
plt.title('Пример визуализации различных параметров эффективности классификаторов')
plt.xlabel('Классификатор')
plt.ylabel('Значение')
plt.legend(loc='best')
plt.show()

# Печать DataFrame с результатами
print(df_results)

# Выбор последнего классификатора (DecisionTreeClassifier)
clf_name = list(classifiers.keys())[-1]
clf = classifiers[clf_name]

# Обучение классификатора и предсказание для тестового набора
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Построение графиков истинных меток, предсказанных меток и ошибок
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolor='k')
plt.title('ожидаемые')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

plt.subplot(1, 3, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k')
plt.title('вычисленные')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

plt.subplot(1, 3, 3)
errors = y_test != y_pred
plt.scatter(X_test[~errors][:, 0], X_test[~errors][:, 1], c='green', label='правильные')
plt.scatter(X_test[errors][:, 0], X_test[errors][:, 1], c='red', label='ошибки')
plt.title('различия')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()

plt.show()

# Вычисление и построение ROC-кривой для классификатора
y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC-кривая {clf_name}')
plt.legend(loc='lower right')
plt.show()

# Генерация сетки для построения границ принятия решения
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Предсказание для всей сетки
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Построение границы принятия решения
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='o', s=20, cmap='coolwarm')
plt.title('Пример визуализации границы принятия решения для разделения на 2 класса')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.show()

# Задание 2

# Определение параметров для GridSearchCV
param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9],
    "p": [1, 2, 3, 4, 5]
}

# Инициализация классификатора и выполнение GridSearchCV
clf = KNeighborsClassifier()
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring="roc_auc")
grid_search.fit(X, y)

# Печать лучших параметров и наилучшего значения ROC AUC из GridSearchCV
print("Лучшие параметры:", grid_search.best_params_)
print("Лучшая AUC:", grid_search.best_score_)

# Извлечение сетки параметров и оценок
param_grid = grid_search.param_grid
n_neighbors = param_grid["n_neighbors"]
p = param_grid["p"]
scores = grid_search.cv_results_["mean_test_score"]
scores = scores.reshape((len(n_neighbors), len(p)))

# Построение тепловой карты оценок
plt.figure(figsize=(10, 6))
plt.imshow(scores, extent=(n_neighbors[0], n_neighbors[-1], p[0], p[-1]), cmap="viridis")
plt.xlabel("Число соседей")
plt.ylabel("P")
plt.title("Влияние параметров n_neighbors и p на AUC")
plt.colorbar()
plt.show()

# Разделение данных на тренировочный и тестовый наборы с 20% тестового набора
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Инициализация классификатора с оптимальными параметрами
clf = KNeighborsClassifier(n_neighbors=5, p=2)

# Обучение и предсказание классификатора
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Печать метрик оценки
print("Точность:", accuracy_score(y_test, y_pred))
print("Чувствительность:", recall_score(y_test, y_pred))
print("Точность:", precision_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_pred))

# Создание DataFrame с результатами
results = {
    "accuracy": accuracy_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "auc": roc_auc_score(y_test, y_pred)
}

df = pd.DataFrame([results])
print(df.to_string())
