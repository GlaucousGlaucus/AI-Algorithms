import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Convert regression target into binary classification
# Above median = 1 (diabetic), Below = 0 (non-diabetic)
median_value = np.median(y)
y = (y > median_value).astype(int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Train model
knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Print correct predictions
print("Correct Predictions:\n")
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")

# Print wrong predictions
print("\nWrong Predictions:\n")
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))