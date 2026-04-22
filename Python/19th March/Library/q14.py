import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.loadtxt("data/Advertising.csv", delimiter=",", skiprows=1)
print(data[:, 0].reshape(-1, 1))
X = data[:, 0].reshape(-1, 1)
y = data[:, -1]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred)
plt.xlabel("TV Advertising")
plt.ylabel("Sales")
plt.show()

print(model.coef_, model.intercept_)
