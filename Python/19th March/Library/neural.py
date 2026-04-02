from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
Y = iris.target

model = MLPClassifier(
    hidden_layer_sizes=(13,),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000
)

model.fit(X, Y)

y_pred = model.predict(X)

print("Accuracy:", round(accuracy_score(Y, y_pred) * 100, 2), "%")