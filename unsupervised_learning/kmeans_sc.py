"""

K-means Clustering Implementations

This code demonstrates implmentations of K-means clustering from scratch and with sklearn.

For the from-scratch implementations:
1. choose the number of clusters K
2. randomly select centroids
3. Assign data points to the nearest cluster
4. Re-initialize centroids
5. Repeat until convergence
- 
"""

import numpy as np
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


# -----------------------------
# 1. K-means Clustering using from-scratch method
# -----------------------------

class KMeans_sc(object):
    def __init__(self, data, k, max_iterations, random_state=42):
        self.data = data
        self.k = k
        self.max_iterations = max_iterations
        rng = np.random.RandomState(random_state)
        self.centroids = data[rng.choice(len(data), k, replace=False)]

    def fit(self):
        for _ in range(self.max_iterations):
            # assign data points to the nearest cluster
            distances = np.linalg.norm(self.data[:, None] - self.centroids, axis=2)
            cluster_assignments = np.argmin(distances, axis=1)

            # update the centroids
            new_centroids = np.array([
                self.data[cluster_assignments == i].mean(axis=0) if np.any(cluster_assignments == i) else self.centroids[i]
                for i in range(self.k)
            ])

            # check for convergence
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

        return cluster_assignments, self.centroids

    def predict(self, new_data):
        distances = np.linalg.norm(new_data[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# -----------------------------
# 2. K-means Clustering using scikit-learn method
# -----------------------------

def Kmeans_sk(data, new_data, n_cluster=3, max_iter=100):
    model = KMeans(n_clusters=n_cluster, max_iter=max_iter, random_state=42).fit(data)
    centroids = model.cluster_centers_
    labels = model.labels_
    return model.predict(new_data)
    

if __name__ == '__main__':
    # Example data (2D points)
    data = np.array([
        [1, 2],
        [1.5, 1.8],
        [5, 8],
        [8, 8],
        [1, 0.6],
        [9, 11],
        [8, 2],
        [10, 2],
        [9, 3]
    ])
    new_data = np.array([[3, 3], [6, 7]])

    # Fit the kmeans model from scratch
    kmeans_sc = KMeans_sc(data, k=3, max_iterations=100)
    cluster_assignments, centroids = kmeans_sc.fit()
    predictions = kmeans_sc.predict(new_data)
    print("Predictions for new data points from scratch model:", predictions)

    # Fit the kmeans model from sklearn
    kmeans_sk = Kmeans_sk(data, new_data, n_cluster=3, max_iter=100)
    print("Predictions for new data points from sklearn model:", kmeans_sk)

