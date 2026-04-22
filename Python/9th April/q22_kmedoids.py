import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn_extra.cluster import KMedoids

iris = load_iris()
X = iris.data

model = KMedoids(n_clusters=3, random_state=42, metric="euclidean")
labels = model.fit_predict(X)
medoids = model.cluster_centers_

colors = ['r', 'g', 'b']

for i in range(3):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i+1}')

plt.scatter(medoids[:, 0], medoids[:, 1], c='black', marker='X', s=200, label='Medoids')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Medoids Clustering (Library)')
plt.legend()
plt.show()
