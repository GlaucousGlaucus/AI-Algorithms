from sklearn.datasets import load_diabetes, load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

diabetes_sklearn = load_iris()
X = diabetes_sklearn.data
Y = diabetes_sklearn.target

Y = (Y > np.mean(Y)).astype(int)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, Y)

y_pred = knn.predict(X)

print("Accuracy:", round(accuracy_score(Y, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(Y, y_pred))
