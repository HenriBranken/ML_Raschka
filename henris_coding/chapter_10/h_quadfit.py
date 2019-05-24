from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

X = np.array([258, 270, 294, 320, 342, 368, 396, 446, 480, 586])[:, np.newaxis]
y = np.array([236.4, 234.4, 252.8, 298.6, 314.2, 342.2, 360.8, 368.0, 391.2,
              390.8])

lr = LinearRegression()
pr = LinearRegression()

quadratic = PolynomialFeatures(degree=2)
X_quad = quadratic.fit_transform(X)

# Fit a simple linear regression model for comparison:
lr.fit(X, y)
X_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(X_fit)

# Fit a multiple regression model on the transformed features for polynomial
# regression
pr.fit(X_quad, y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

# Plot the results
plt.scatter(X, y, label="Training Points", marker="o", c="steelblue",
            edgecolor="linen")
plt.plot(X_fit, y_lin_fit, label="Linear Fit", ls="--", c="indianred")
plt.plot(X_fit, y_quad_fit, label="Quadratic Fit", c="seagreen")
plt.legend(loc="upper left")
plt.grid()
plt.tight_layout()
plt.show()

y_lin_pred = lr.predict(X)
y_quad_pred = pr.predict(X_quad)
print("Training MSE Linear: "
      "{:.3f}.".format(mean_squared_error(y_true=y, y_pred=y_lin_pred)))
print("Training MSE Quadratic: "
      "{:.3f}.".format(mean_squared_error(y_true=y, y_pred=y_quad_pred)))
print("Training R^2 Linear: "
      "{:.3f}.".format(r2_score(y_true=y, y_pred=y_lin_pred)))
print("Training R^2 Quadratic: "
      "{:.3f}.".format(r2_score(y_true=y, y_pred=y_quad_pred)))
