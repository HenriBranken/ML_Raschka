from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt


def lin_regplot(x_vals, y_vals, model):
    plt.scatter(x_vals, y_vals, c="steelblue", edgecolor="aliceblue", s=70)
    plt.plot(x_vals, model.predict(x_vals), c="k", lw=2)
    return None


df = pd.read_csv("./housing_data.txt", sep="\s+", header=None)
df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
X = df[["LSTAT"]].values
y = df["MEDV"].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort()

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel("% lower status of the population [LSTAT]")
plt.ylabel("Price in $1000s [MEDV]")
plt.grid()
plt.tight_layout()
plt.show()
