from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score


class KNN(neighbors.KNeighborsClassifier):

    def __init__(self, n_neighbors=1, use_kd_tree=False):
        super().__init__(n_neighbors=n_neighbors, algorithm='auto' if use_kd_tree else 'ball_tree')
        self.use_kd_tree = use_kd_tree

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)

    def visualize_decision_boundary(self, X, y, resolution=100):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("KNN Decision Boundary")
        plt.show()

    def visualize_data_3d(self, X, y):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with color based on class label
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=30)

        # Set axis labels
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")

        # Set axis limits slightly extended from data range
        ax.set_xlim3d(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ax.set_ylim3d(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
        ax.set_zlim3d(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

        # Set title
        ax.set_title("3D Data Visualization with Class Labels")
        plt.show()

    def evaluate_with_cross_validation(self, X, y, cv=5):
        scores = cross_val_score(self, X, y, cv=cv, scoring='accuracy')
        return scores.mean(), scores.std()


X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=3
)

knn = KNN(n_neighbors=5)

knn.fit(X, y)

new_data = np.array([[1.5, 2.0]])
predicted_label = knn.predict(new_data)
print(f"Predicted label for new data point: {predicted_label}")

accuracy, std = knn.evaluate_with_cross_validation(X, y)
print(f"Average accuracy: {accuracy:.4f} +- {std:.4f}")

knn.visualize_decision_boundary(X, y)
knn.visualize_data_3d(X, y)


class KNNRegressor(neighbors.KNeighborsRegressor):

    def __init__(self, n_neighbors=1):
        super().__init__(n_neighbors=n_neighbors)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)

    def score(self, X, y):
        return super().score(X, y)

    def visualize_regression_line(self, X, y, resolution=100):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx = np.linspace(x_min, x_max, resolution)
        yy = self.predict(xx.reshape(-1, 1))

        plt.figure()
        plt.scatter(X, y, c='b', s=10, label='Data points')
        plt.plot(xx, yy, label='Regression line')
        plt.xlabel("Feature")
        plt.ylabel("Target")
        plt.title("KNN Regression Line")
        plt.legend()
        plt.show()

    def evaluate_with_cross_validation(self, X, y, cv=5):
        scores = cross_val_score(self, X, y, cv=cv, scoring='neg_mean_squared_error')
        return np.mean(-scores), np.std(-scores)


X, y = make_regression(
    n_samples=100,
    n_features=1,
    n_informative=1,
    noise=20,
    random_state=4
)

knn_reg = KNNRegressor(n_neighbors=5)
knn_reg.fit(X, y)

knn_reg.visualize_regression_line(X, y)

mse, mse_std = knn_reg.evaluate_with_cross_validation(X, y)
print(f"Mean Squared Error (MSE): {mse:.2f} +- {mse_std:.2f}")
