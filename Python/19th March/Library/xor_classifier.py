from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
Y = [
    0, 1, 1, 0
]

model = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation="tanh",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1000
)

model.fit(X, Y)

y_pred = model.predict(X)

print("Accuracy:", round(accuracy_score(Y, y_pred) * 100, 2), "%")
