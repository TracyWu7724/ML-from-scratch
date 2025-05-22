# -*- coding = utf-8 -*-
# @Time : 5/7/25 18:10
# @Author : Tracy
# @File : kmeans_sc.py
# @Software : PyCharm


import numpy as np
from sklearn.cluster import KMeans

class KMeans_sc(object):
    """
    K-Means clustering from sractch
    1. choose the number of clusters K
    2. randomly select centroids
    3. Assign data points to the nearest cluster
    4. Re-initialize centroids
    5. Repeat until convergence
    """
    def __init__(self, data, k, max_iterations, random_state=42):
        # Step 1: Initialize the K value and max_iters
        self.data = data
        self.k = k
        self.max_iterations = max_iterations

        # Step 2: Randomly choose initial centroids from the data points
        rng = np.random.RandomState(random_state)
        self.centroids = data[rng.choice(len(data), k, replace=False)]

    def fit(self):
        for _ in range(self.max_iterations):
            distances = np.zeros((len(self.data), self.k))

            # Step 3: Assign data points to the nearest cluster
            for i in range(len(self.data)):
                for j in range(self.k):
                    distances[i, j] = np.sqrt(np.sum((self.data[i] - self.centroids[j])**2))
            cluster_assignments = np.argmin(distances, axis=1)

            # Step 4: Recalculate the centroids
            new_centroids = np.zeros((self.k, self.data.shape[1]))
            for m in range(self.k):
                cluster_points = self.data[cluster_assignments == m]

                # Update centroid to the mean of the points in the cluster
                if len(cluster_points) > 0:
                    new_centroids[m] = np.mean(cluster_points, axis=0)
                else:
                    # Keep the old centroid if no points are assigned
                    new_centroids[m] = self.centroids[m]

            # Step 5: Check for convergence
            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids

        return cluster_assignments, self.centroids

    def predict(self, new_data):
        """
        Predict the cluster labels for new data points based on the trained centroids.
        """
        distances = np.zeros((len(new_data), self.k))

        # Calculate the distance from each new data point to each centroid
        for i in range(len(new_data)):
            for j in range(self.k):
                distances[i, j] = np.sqrt(np.sum((new_data[i] - self.centroids[j])**2))

        # Assign each new point to the closest centroid
        cluster_assignments = np.argmin(distances, axis=1)

        return cluster_assignments

class Kmeans_sk(object):
    def __init__(self, data, k, max_iterations, random_state=42):
        # Initialize KMeans from sklearn with the given parameters
        self.Kmeans = KMeans(n_clusters=k, max_iter=max_iterations, random_state=random_state)
        self.Kmeans.fit(data)  # Fit the model on the data

        # Store the results
        self.centroids = self.Kmeans.cluster_centers_
        self.labels = self.Kmeans.labels_

    def get_centroids(self):
        return self.centroids

    def get_labels(self):
        return self.labels

    def predict(self, new_data):
        return self.Kmeans.predict(new_data)


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

    # Fit the kmeans model from scratch
    kmeans_sc = KMeans_sc(data, k=3, max_iterations=100)
    cluster_assignments, centroids = kmeans_sc.fit()
    print("Cluster Assignments:", cluster_assignments)
    print("Centroids:", centroids)

    # Predict cluster assignments for new data points using the kmeans model from scratch
    new_data = np.array([[3, 3], [6, 7]])
    predictions = kmeans_sc.predict(new_data)
    print("Predictions for new data points from scratch model:", predictions)

    # Fit the kmeans model from sklearn
    kmeans_sk = Kmeans_sk(data, k=3, max_iterations=100)
    cluster_assignments = kmeans_sk.get_labels()
    centroids = kmeans_sk.get_centroids()
    print("Cluster Assignments:", cluster_assignments)
    print("Centroids:", centroids)

    # Predict cluster assignments for new data points using the kmeans model from sklearn
    predictions = kmeans_sk.predict(new_data)
    print("Predictions for new data points from sklearn model:", predictions)

    print("When the random state and all other parameters are set to identical values, the results will be consistent.")
