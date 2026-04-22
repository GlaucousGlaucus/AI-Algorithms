import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()
X = iris.data
Y = iris.target

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_counts = np.round(X_scaled * 10).astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_counts, Y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("MultinomialNB Accuracy:", round(accuracy_score(Y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, y_pred))
