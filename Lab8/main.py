import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# zadanie 1
def distp(X, C, e):
    X_norm = np.linalg.norm(X, axis=1)[:, np.newaxis]
    C_norm = np.linalg.norm(C, axis=1)[:, np.newaxis]
    # Reshape C to match the number of rows in X
    C = C.reshape((X.shape[0], C.shape[1])) # вроде он тут доебался до чего то хуй ве, уже не помню
    X_C = X - C
    return np.sqrt(X_C ** 2 + (X_norm ** 2 - 2 * np.dot(X, C.T) + C_norm ** 2))



def distm(X, C, V):
    X_C = X - C
    return np.sqrt(np.einsum('ij,jk,kl->il', X_C, np.linalg.inv(V), X_C)) # это нахуй переделать, бо я не еду что делает этахуета с буквами


def kmeans(X, k, e=1e-4, max_iter=100):
    n_samples, n_features = X.shape
    C = X[np.random.choice(n_samples, k, replace=False), :]  # Initialize centroids randomly
    CX = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        old_CX = CX.copy()
        distances = distp(X, C, e)
        CX = np.argmin(distances, axis=1)
        for i in range(k):
            C[i] = np.mean(X[CX == i], axis=0)
        if np.all(old_CX == CX):
            break
    return C, CX


# zadanie 2
df = pd.read_csv("autos.csv")

categorical_columns = [
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "engine-type",
    "num-of-cylinders",
    "fuel-system",
]
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

features = [
    "normalized-losses",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-size",
    "horsepower",
    "city-mpg",
    "highway-mpg",
]
X = df[features].to_numpy()

k = 3
C, CX = kmeans(X, k)

df["cluster"] = CX

print(df)

# zadanie 3

features_to_plot = ["horsepower", "city-mpg"]

X_plot = df[features_to_plot]
cluster_labels = df["cluster"]

plt.figure(figsize=(10, 6))
for i in range(k):
    cluster_data = X_plot[cluster_labels == i]
    plt.scatter(cluster_data["horsepower"], cluster_data["city-mpg"], label=f"Cluster {i + 1}")

plt.title("Car Clusters by Horsepower and City MPG")
plt.xlabel("Horsepower")
plt.ylabel("City MPG")
plt.legend()
plt.grid(True)
plt.show()


# zadanie 4
def silhouette_coefficient(X, CX):
    n_samples, n_features = X.shape
    k = np.unique(CX).shape[0]

    intra_cluster_dist = np.zeros((n_samples, k))
    for i in range(n_samples):
        cluster_i = CX[i]
        cluster_data_i = X[CX == cluster_i]
        for j in range(len(cluster_data_i)):
            dist = np.linalg.norm(X[i] - cluster_data_i[j])
            intra_cluster_dist[i, cluster_i] += dist
        intra_cluster_dist[i, cluster_i] /= (max(0, len(cluster_data_i) - 1))

    inter_cluster_dist = np.zeros((n_samples, k))
    for i in range(n_samples):
        cluster_i = CX[i]
        for j in range(k):
            if j == cluster_i:
                continue
            cluster_data_j = X[CX == j]
            for p in range(len(cluster_data_j)):
                dist = np.linalg.norm(X[i] - cluster_data_j[p])
                inter_cluster_dist[i, j] = min(inter_cluster_dist[i, j], dist)

    silhouette = np.zeros(n_samples)
    for i in range(n_samples):
        if intra_cluster_dist[i, CX[i]] == 0:
            silhouette[i] = 0
        else:
            silhouette[i] = (inter_cluster_dist[i, CX[i]] - intra_cluster_dist[i, CX[i]]) / max(
                intra_cluster_dist[i, CX[i]], inter_cluster_dist[i, CX[i]]
            )
    return silhouette


silhouette_vals = silhouette_coefficient(X, CX)

print(
    "result:", np.mean(silhouette_vals)
)
