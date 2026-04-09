import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

# Generate two semicircles dataset
X, y = make_moons(n_samples=300, noise=0.05, random_state=0)

# K-Means
kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
kmeans_labels = kmeans.fit_predict(X)

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Plot original data
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Original Data")

# Plot K-Means result
plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.title("K-Means Clustering")

# Plot DBSCAN result
plt.subplot(1, 3, 3)
plt.scatter(X[:, 0], X[:, 1], c=dbscan_labels)
plt.title("DBSCAN Clustering")

plt.tight_layout()
plt.show()
