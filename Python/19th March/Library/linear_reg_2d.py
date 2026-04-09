import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.show()