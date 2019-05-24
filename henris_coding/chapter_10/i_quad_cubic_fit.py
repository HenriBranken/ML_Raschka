from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./housing_data.txt", sep="\s+", header=None)

df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

X = df[["LSTAT"]].values
y = df["MEDV"].values

regr = LinearRegression()

# Create higher-order features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)

# Generate the higher-order feature matrices
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Generate linearly-spaced X values
X_fit = np.arange(X.min() - 1, X.max() + 1, 1)[:, np.newaxis]

# Fit the features:
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
r2_linear = r2_score(y_true=y, y_pred=regr.predict(X))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
r2_quad = r2_score(y_true=y, y_pred=regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
r2_cubic = r2_score(y_true=y, y_pred=regr.predict(X_cubic))

# Plot the results
plt.scatter(X, y, label="Training Data", c="silver", edgecolor="silver")

plt.plot(X_fit, y_lin_fit,
         label="Linear (d=1), R^2 = {:.3f}".format(r2_linear), color="blue",
         lw=2, linestyle=":")

plt.plot(X_fit, y_quad_fit,
         label="Quadratic (d=2), R^2 = {:.3f}".format(r2_quad), color="red",
         lw=2, linestyle="-")

plt.plot(X_fit, y_cubic_fit,
         label="Cubic (d=3), R^2 = {:.3f}".format(r2_cubic), color="green",
         lw=2, linestyle="--")

plt.plot(X_fit, (-1.295*np.log(X_fit) + 7.724)**2, c="burlywood", lw=2)

plt.xlabel("% Lower Status of the population [LSTAT]")
plt.ylabel("Price in $1000s [MEDV]")
plt.legend(loc="upper right")
plt.grid()
plt.tight_layout()
plt.show()
