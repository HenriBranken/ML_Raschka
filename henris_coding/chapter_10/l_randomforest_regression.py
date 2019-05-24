from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./housing_data.txt", sep="\s+", header=None)

df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

X = df.iloc[:, :-1].values
y = df["MEDV"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=1)

forest = RandomForestRegressor(n_estimators=1000, criterion="mse",
                               random_state=1, n_jobs=-1)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
print("MSE train: {:.3f}.".format(mean_squared_error(y_true=y_train,
                                                     y_pred=y_train_pred)))
print("MSE test: {:.3f}.".format(mean_squared_error(y_true=y_test,
                                                    y_pred=y_test_pred)))
print("R^2 train: {:.3f}.".format(r2_score(y_true=y_train,
                                           y_pred=y_train_pred)))
print("R^2 test: {:.3f}.".format(r2_score(y_true=y_test,
                                          y_pred=y_test_pred)))

# Let us take a look at the residuals of the prediction.
plt.scatter(y_train_pred, y_train_pred - y_train, c="steelblue",
            edgecolor="white", marker="o", s=35, alpha=0.7,
            label="Training Data")
plt.scatter(y_test_pred, y_test_pred - y_test, c="seagreen", edgecolor="ivory",
            marker="^", s=35, alpha=0.7, label="Test Data")
plt.axhline(y=0, lw=2, color="k")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.xlim(left=-10, right=50)
plt.grid()
plt.tight_layout()
plt.show()
