import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN

# Load dataset
iris = load_iris()
X = iris.data

# DBSCAN model
model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X)

# Visualization (first 2 features)
unique_labels = set(labels)
colors = ['r', 'g', 'b', 'y', 'c']

for label in unique_labels:
    if label == -1:
        color = 'black'  # noise
    else:
        color = colors[label % len(colors)]

    cluster = X[labels == label]
    plt.scatter(cluster[:, 0], cluster[:, 1], c=color, label=f'Cluster {label}')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('DBSCAN Clustering on Iris Dataset')
plt.legend()
plt.show()