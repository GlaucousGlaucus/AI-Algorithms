import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data

K = 3

# Initialize medoids randomly (actual data points)
np.random.seed(42)
random_indices = np.random.choice(len(X), K, replace=False)
medoids = X[random_indices]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

for iteration in range(100):
    clusters = [[] for _ in range(K)]

    for point in X:
        distances = [euclidean_distance(point, medoid) for medoid in medoids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    clusters = [np.array(cluster) for cluster in clusters]

    new_medoids = []

    for cluster in clusters:
        distances_sum = []
        for candidate in cluster:
            total_distance = np.sum([euclidean_distance(candidate, other) for other in cluster])
            distances_sum.append(total_distance)

        # Choose point with minimum total distance
        new_medoids.append(cluster[np.argmin(distances_sum)])

    new_medoids = np.array(new_medoids)

    # Check convergence
    if np.all(medoids == new_medoids):
        break

    medoids = new_medoids

# Visualization (same as yours)
colors = ['r', 'g', 'b']

for i, cluster in enumerate(clusters):
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i+1}')

plt.scatter(medoids[:, 0], medoids[:, 1], c='black', marker='X', s=200, label='Medoids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Medoids Clustering on Iris Dataset')
plt.legend()
plt.show()
