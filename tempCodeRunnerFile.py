kmeans_sc = KMeans_sc(data, k=3, max_iterations=100)
cluster_assignments, centroids = kmeans_sc.fit()
predictions = kmeans_sc.predict(new_data)
print("Predictions for new data points from scratch model:", predictions)