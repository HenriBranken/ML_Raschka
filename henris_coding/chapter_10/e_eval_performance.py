from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./housing_data.txt", sep="\s+", header=None)

df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

X = df.iloc[:, :-1].values
y = df["MEDV"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

slr = LinearRegression()
slr.fit(X, y)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(y_train_pred, y_train_pred - y_train, c="steelblue", marker="o",
            edgecolor="white", label="Training Data")
plt.scatter(y_test_pred, y_test_pred - y_test, c="salmon", marker="X",
            edgecolor="linen", label="Test Data")
plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.legend(loc="upper left")
plt.hlines(y=0, xmin=-10, xmax=50, color="k", lw=2)
plt.xlim(left=-10, right=50)
plt.show()
