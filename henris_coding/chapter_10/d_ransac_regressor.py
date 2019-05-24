from sklearn.linear_model import RANSACRegressor, LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("./housing_data.txt", sep="\s+", header=None)

df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

X = df[["RM"]].values
y = df["MEDV"].values

ransac = RANSACRegressor(base_estimator=LinearRegression(),
                         max_trials=1000,
                         min_samples=0.75,
                         loss="absolute_loss",
                         residual_threshold=5.0,
                         random_state=0)

ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
X_line = np.arange(3, 10, 1)
y_line_ransac = ransac.predict(X_line[:, np.newaxis])

plt.scatter(X[inlier_mask], y[inlier_mask], c="steelblue", edgecolor="white",
            marker="o", label="Inliers")
plt.scatter(X[outlier_mask], y[outlier_mask], c="limegreen", edgecolor="white",
            marker="X", label="Outliers")
plt.plot(X_line, y_line_ransac, color="black", lw=2)
plt.xlabel("Average number of rooms [RM]")
plt.ylabel("Price in $1000s [MEDV]")
plt.legend(loc="upper left")
plt.show()

print("Slope: {:.3f}.".format(ransac.estimator_.coef_[0]))
print("Intercept: {:.3f}.".format(ransac.estimator_.intercept_))
