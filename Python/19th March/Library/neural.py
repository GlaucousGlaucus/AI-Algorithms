from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
Y = iris.target

model = MLPClassifier(
    hidden_layer_sizes=(9,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.5,
    max_iter=500
)

model.fit(X, Y)

y_pred = model.predict(X)

print("Accuracy:", round(accuracy_score(Y, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(Y, y_pred))