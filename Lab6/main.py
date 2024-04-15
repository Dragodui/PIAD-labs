import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def wiPCA(X, n_components=1):
    mean_vec = np.mean(X, axis=0)
    X_centered = X - mean_vec

    cov_mat = np.cov(X_centered.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    eigenvectors = eigenvectors[:, :n_components]

    X_reduced = X_centered @ eigenvectors

    return X_reduced, eigenvalues, eigenvectors, mean_vec


# zadanie 1
def zadanie1():
    X = np.random.rand(200, 2)

    X_reduced, eigenvalues, eigenvectors, mean_vec = wiPCA(X, n_components=1)

    plt.scatter(X[:, 0], X[:, 1])
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Original Data")
    plt.show()

    plt.arrow(mean_vec[0], mean_vec[1], eigenvectors[:, 0][0], eigenvectors[:, 0][1], color='red', label='PC1')
    plt.scatter(X_reduced, np.zeros(X_reduced.shape), alpha=0.4)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Transformed Data with PC1")
    plt.legend()
    plt.show()


# zadanie 2
def zadanie2():
    iris = datasets.load_iris()
    X = iris.data

    X_reduced, eigenvalues, eigenvectors, mean_vec = wiPCA(X, n_components=2)

    plt.scatter(X[:, 0], X[:, 1], c=iris.target, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Original Iris Data")
    plt.show()

    plt.arrow(mean_vec[0], mean_vec[1], eigenvectors[:, 0][0], eigenvectors[:, 0][1], color='red', label='PC1')
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.4)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Transformed Iris Data with PC1")
    plt.legend()
    plt.show()


# zadanie3
def zadanie3():
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target
    X_reduced_2d, eigenvalues, eigenvectors, mean_vec = wiPCA(X, n_components=2)

    plot_explained_variance(eigenvalues)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("Original Digits Data")
    plt.show()

    plt.scatter(X_reduced_2d[:, 0], X_reduced_2d[:, 1], c=y, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Transformed Digits Data (Reduced to 2 Dimensions)")
    plt.show()

    errors = reconstruction_error(X, X_reduced_2d, eigenvectors[:, :2], mean_vec)
    print("Average reconstruction error with 2 principal components:", np.mean(errors))


def plot_explained_variance(eigenvalues):
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance

    plt.plot(explained_variance_ratio)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance Ratio vs Number of Principal Components")
    plt.show()


def reconstruction_error(X, X_reduced, eigenvectors, mean_vec):
    X_reconstructed = X_reduced @ eigenvectors.T + mean_vec
    errors = np.sum((X - X_reconstructed) ** 2, axis=1)
    return errors

zadanie1()
# zadanie2()
# zadanie3()

