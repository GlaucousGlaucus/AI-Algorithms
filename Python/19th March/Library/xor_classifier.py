from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
Y = [0, 1, 1, 0]

model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="tanh",
    solver="adam",
    max_iter=10000,
    random_state=42
)

model.fit(X, Y)

y_pred = model.predict(X)

print("Predictions:", y_pred)
print("Accuracy:", round(accuracy_score(Y, y_pred) * 100, 2), "%")
