import numpy as np
from scipy.stats import entropy
import pandas as pd
from scipy.sparse import csr_matrix


# zadanie 1
def freq(x, prob=True):
    unique_values, counts = np.unique(x, return_counts=True)

    if prob:
        ni = counts / np.sum(counts)
    else:
        ni = counts

    return unique_values, ni


# zadanie 2
def freq2(x, y, prob=True):
    unique_values_x, counts_x = np.unique(x, return_counts=True)
    unique_values_y, counts_y = np.unique(y, return_counts=True)

    if prob:
        ni = np.outer(counts_x, counts_y) / np.sum(counts_x) / np.sum(counts_y)
    else:
        ni = np.outer(counts_x, counts_y)

    return unique_values_x, unique_values_y, ni


# zadanie 3
def entropy(x):
    unique_values, counts = np.unique(x, return_counts=True)
    probabilities = counts / np.sum(counts)
    return -np.sum(probabilities * np.log2(probabilities))


def entropy_cond(x, y):
    unique_y, counts_y = np.unique(y, return_counts=True)
    entropy_cond = 0
    for i, (value_y, count_y) in enumerate(zip(unique_y, counts_y)):
        x_filtered = x[y == value_y]
        entropy_cond += count_y / np.sum(counts_y) * entropy(x_filtered)
    return entropy_cond


def information_gain(x, y):
    H_X = entropy(x)
    H_X_given_Y = entropy_cond(x, y)
    return H_X - H_X_given_Y


# zadanie 4
def select_features(data, target_attribute):
    information_gains = {}
    for feature in data.columns:
        if feature != target_attribute:
            information_gains[feature] = information_gain(data[target_attribute], data[feature])
    selected_features = sorted(information_gains, key=information_gains.get, reverse=True)
    return selected_features


# Load data
data = pd.read_csv("zoo.csv")

target_attribute = "type"

selected_features = select_features(data.copy(), target_attribute)

print(selected_features)


# zadanie 5
def freq_sparse(X_sparse, prob=True):
    unique_values = []
    counts = []
    for i in range(X_sparse.shape[0]):
        for j in range(X_sparse.shape[1]):
            if X_sparse[i, j] != 0:
                if X_sparse[i, j] not in unique_values:
                    unique_values.append(X_sparse[i, j])
                    counts.append(1)
                else:
                    counts[unique_values.index(X_sparse[i, j])] += 1

    if prob:
        ni = np.array(counts) / np.sum(counts)
    else:
        ni = np.array(counts)
    return unique_values, ni


def freq2_sparse(X_sparse1, X_sparse2, prob=True):
    unique_values_x = []
    unique_values_y = []
    counts = []
    for i in range(X_sparse1.shape[0]):
        for j in range(X_sparse1.shape[1]):
            if X_sparse1[i, j] != 0:
                for k in range(X_sparse2.shape[1]):
                    if X_sparse2[i, k] != 0:
                        if (X_sparse1[i, j], X_sparse2[i, k]) not in zip(unique_values_x, unique_values_y):
                            unique_values_x.append(X_sparse1[i, j])
                            unique_values_y.append(X_sparse2[i, k])
                            counts.append(1)
                        else:
                            counts[zip(unique_values_x, unique_values_y).index((X_sparse1[i, j], X_sparse2[i, k]))] += 1

    if prob:
        ni = np.outer(counts, counts) / np.sum(counts) / np.sum(counts)
    else:
        ni = np.outer(counts, counts)
    return ni


# zadanie 6
