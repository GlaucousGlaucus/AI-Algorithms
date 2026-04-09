from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
Y = iris.target

model = LogisticRegression(max_iter=200)

model.fit(X, Y)

y_pred = model.predict(X)

print("Accuracy:", round(accuracy_score(Y, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(Y, y_pred))
