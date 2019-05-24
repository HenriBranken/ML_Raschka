from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd


def lin_regplot(X, y, model):
    plt.scatter(X, y, c="steelblue", edgecolor="mistyrose", s=70)
    plt.plot(X, model.predict(X), c="k", lw=2)
    return None


df = pd.read_csv("./housing_data.txt", sep="\s+", header=None)

df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

X = df[["RM"]].values
y = df["MEDV"].values

slr = LinearRegression()
slr.fit(X, y)
print("Slope: {:.3f}.".format(slr.coef_[0]))
print("Intercept: {:.3f}.".format(slr.intercept_))

lin_regplot(X, y, slr)
plt.xlabel("Average number of Rooms [RM] (standardized)")
plt.ylabel("Price in $1000s [MEDV] (standardized)")
plt.show()
