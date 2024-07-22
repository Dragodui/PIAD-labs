# Импортируем необходимые библиотеки и модули для кластеризации и анализа данных
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.stats import mode
import scipy
import matplotlib.image as mpimg
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error
import cv2

# Кластеризация
#1 Загружаем набор данных Iris
iris = datasets.load_iris()
X = iris.data  # Функции (признаки) цветов
Y = iris.target  # Истинные метки классов

#2 Пробуем разные методы агломеративной кластеризации
linkages = ['ward', 'complete', 'average', 'single']  # Типы линкирования
agglomerative_results = {}  # Словарь для хранения результатов кластеризации

for linkage in linkages:
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=3)  # Инициализируем модель
    Y_pred = clustering.fit_predict(X)  # Обучаем модель и предсказываем кластеры
    agglomerative_results[linkage] = Y_pred  # Сохраняем предсказания

#3 Функция для нахождения соответствия между предсказанными и истинными метками классов
def find_perm(clusters, Y_real, Y_pred):
    perm = []
    for i in range(clusters):
        idx = Y_pred == i  # Индексы элементов, отнесенных к кластеру i
        new_label = scipy.stats.mode(Y_real[idx])[0]  # Определяем наиболее частую метку в этом кластере
        perm.append(new_label)  # Сохраняем эту метку
    return np.array([perm[label] for label in Y_pred])  # Переназначаем метки кластеров

# Переназначаем метки кластеров для всех методов линкирования
Y_pred_mapped = {linkage: find_perm(3, Y, Y_pred) for linkage, Y_pred in agglomerative_results.items()}

#4 Вычисляем коэффициенты Жаккара для каждого метода линкирования
jaccard_indices = {linkage: jaccard_score(Y, Y_pred_mapped[linkage], average='macro') for linkage in linkages}

#5 Функция для визуализации кластеров
def plot_clusters(X_reduced, Y, Y_pred, title, ax):
    colors = ['purple', 'yellow', 'green']
    markers = ['o', 's', 'D']
    for i, color, marker in zip(np.unique(Y), colors, markers):
        points = X_reduced[Y == i]  # Точки, принадлежащие классу i
        hull = ConvexHull(points)  # Построение выпуклой оболочки
        ax.scatter(points[:, 0], points[:, 1], label=f'Class {i}', c=color, marker=marker)  # Визуализация точек
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], color)  # Отрисовка граней выпуклой оболочки
    ax.set_title(title)  # Установка заголовка графика
    ax.legend()  # Добавление легенды

# Применяем PCA для снижения размерности до 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Создаем фигуру и оси для 2D визуализации
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Визуализируем исходные классы
plot_clusters(X_reduced, Y, Y, 'Original Classes', axs[0])

# Визуализируем кластеры, полученные методом линкирования 'ward'
plot_clusters(X_reduced, Y_pred_mapped['ward'], Y_pred_mapped['ward'], 'Clustered Classes (Ward)', axs[1])

# Визуализируем корректные и некорректные предсказания
correct = Y == Y_pred_mapped['ward']
incorrect = ~correct
axs[2].scatter(X_reduced[correct, 0], X_reduced[correct, 1], c='green', label='Correct')  # Корректные предсказания
axs[2].scatter(X_reduced[incorrect, 0], X_reduced[incorrect, 1], c='red', label='Incorrect')  # Некорректные предсказания
axs[2].set_title('Differences')  # Заголовок графика
axs[2].legend()  # Легенда

# Показываем графики
plt.show()

#6 3D визуализация

# Применяем PCA для снижения размерности до 3D
pca = PCA(n_components=3)
X_reduced_3d = pca.fit_transform(X)

# Создаем фигуру для 3D визуализации
fig = plt.figure(figsize=(15, 5))

# Визуализируем исходные классы в 3D
ax = fig.add_subplot(131, projection='3d')
plot_clusters(X_reduced_3d, Y, Y, 'Original Classes', ax)

# Визуализируем кластеры, полученные методом линкирования 'ward', в 3D
ax = fig.add_subplot(132, projection='3d')
plot_clusters(X_reduced_3d, Y_pred_mapped['ward'], Y_pred_mapped['ward'], 'Clustered Classes (Ward)', ax)

# Визуализируем корректные и некорректные предсказания в 3D
ax = fig.add_subplot(133, projection='3d')
correct = Y == Y_pred_mapped['ward']
incorrect = ~correct
ax.scatter(X_reduced_3d[correct, 0], X_reduced_3d[correct, 1], X_reduced_3d[correct, 2], c='green', label='Correct')  # Корректные предсказания
ax.scatter(X_reduced_3d[incorrect, 0], X_reduced_3d[incorrect, 1], X_reduced_3d[incorrect, 2], c='red', label='Incorrect')  # Некорректные предсказания
ax.set_title('Differences')  # Заголовок графика
ax.legend()  # Легенда

# Показываем графики
plt.show()

#7 Построение дендрограммы

# Вычисляем связки (linkage) для метода 'ward'
linked = linkage(X, 'ward')

# Визуализируем дендрограмму
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=Y)
plt.title('Dendrogram (Ward)')
plt.show()

#8 Сравнение с другими методами кластеризации

# Метод K-Means
kmeans = KMeans(n_clusters=3)
Y_kmeans = kmeans.fit_predict(X)
Y_kmeans_mapped = find_perm(3, Y, Y_kmeans)

# Метод Gaussian Mixture
gmm = GaussianMixture(n_components=3)
Y_gmm = gmm.fit_predict(X)
Y_gmm_mapped = find_perm(3, Y, Y_gmm)


#9 Загружаем и обрабатываем данные для зоопарка

zoo_data = pd.read_csv('zoo.csv')
X_zoo = zoo_data.drop(columns=['animal', 'type']).values  # Удаляем столбцы с названиями животных и типами
Y_zoo = zoo_data['type'].values  # Выделяем столбец с типами животных

# Квантование

#1 Загружаем изображение
image = mpimg.imread("image.jpg")

#2 Сохраняем оригинальное изображение
original_image = image.copy()

#3 Преобразуем изображение в двумерный массив
image_data = image.reshape(307200, 3)

#4 Определяем количество кластеров для квантования
cluster_counts = [2, 3, 5, 10, 30, 100]

#5 Применяем различные методы кластеризации для квантования изображения
clustering_methods = [
    ("K-Means", KMeans),
    ("Gaussian Mixture", GaussianMixture)
]

# Проходим по всем методам и количествам кластеров
for method_name, clustering_class in clustering_methods:
    for cluster_count in cluster_counts:
        if clustering_class == AgglomerativeClustering:
            pass  # Пропускаем метод AgglomerativeClustering
        elif clustering_class == KMeans:
            clusterer = clustering_class(n_clusters=cluster_count, n_init=10)  # Инициализируем K-Means кластеризатор
            cluster_labels = clusterer.fit_predict(image_data)  # Предсказываем кластеры для пикселей изображения
            cluster_centers = clusterer.cluster_centers_  # Получаем центры кластеров
        else:
            clusterer = clustering_class(n_components=cluster_count)  # Инициализируем Gaussian Mixture кластеризатор
            cluster_labels = clusterer.fit_predict(image_data)  # Предсказываем кластеры для пикселей изображения
            cluster_centers = clusterer.means_  # Получаем центры кластеров

        # Создаем квантованное изображение
        quantized_image = np.zeros_like(image_data)
        for i in range(cluster_count):
            quantized_image[cluster_labels == i] = cluster_centers[i]  # Присваиваем пикселям цвета центров кластеров
