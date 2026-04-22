import numpy as np
import matplotlib.pyplot as plt

def lwlr(X, y, tau):
    m = X.shape[0]
    X_bias = np.c_[np.ones(m), X]
    y_pred = []

    for i in range(m):
        x_query = X[i]
        W = np.eye(m)

        for j in range(m):
            diff = X[j] - x_query
            W[j, j] = np.exp(-(diff @ diff) / (2 * tau**2))

        theta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ (X_bias.T @ W @ y)
        y_pred.append(np.array([1, x_query[0]]) @ theta)

    return np.array(y_pred)

np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 100)

tau = 0.5
y_pred = lwlr(X, y, tau)

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Locally Weighted Regression")
plt.show()