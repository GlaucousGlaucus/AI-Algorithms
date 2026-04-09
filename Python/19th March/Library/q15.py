from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()
X, y = data.data, data.target

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

print(model.predict(X[:5]))

plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True
)
plt.show()
