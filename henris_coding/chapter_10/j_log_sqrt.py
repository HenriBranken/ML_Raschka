from sklearn.linear_model import LinearRegression
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

# Transform the features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# Fit the features
regr = regr.fit(X_log, y_sqrt)
X_fit = np.arange(X_log.min() - 1, X_log.max() + 1, 1)[:, np.newaxis]
y_lin_fit = regr.predict(X_fit)
r2_linear = r2_score(y_true=y_sqrt, y_pred=regr.predict(X_log))

print("regr.coef_ = {}.".format(regr.coef_))
print("regr.intercept_ = {}.".format(regr.intercept_))

# Plot the results
plt.scatter(X_log, y_sqrt, label="Training Points", color="silver",
            edgecolor="silver")
plt.plot(X_fit, y_lin_fit,
         label="Linear (d=1), R^2 = {:.3f}".format(r2_linear), c="blue", lw=2)
plt.xlabel("log(% LSTAT)")
plt.ylabel("sqrt([MEDV])")
plt.legend(loc="lower left")
plt.tight_layout()
plt.grid()
plt.show()
