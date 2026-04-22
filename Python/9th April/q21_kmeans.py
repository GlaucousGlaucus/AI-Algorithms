import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data  # features

# Number of clusters
K = 3

# Step 1: Initialize centroids randomly
np.random.seed(42)
random_indices = np.random.choice(len(X), K, replace=False)
centroids = X[random_indices]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# K-Means algorithm
for iteration in range(100):
    clusters = [[] for _ in range(K)]

    # Step 2: Assign points to nearest centroid
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    # Convert clusters to numpy arrays
    clusters = [np.array(cluster) for cluster in clusters]

    # Step 3: Update centroids
    new_centroids = np.array([cluster.mean(axis=0) for cluster in clusters])

    # Check for convergence
    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

# Step 4: Visualization (2D using first two features)
colors = ['r', 'g', 'b']

for i, cluster in enumerate(clusters):
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Means Clustering on Iris Dataset')
plt.legend()
plt.show()
