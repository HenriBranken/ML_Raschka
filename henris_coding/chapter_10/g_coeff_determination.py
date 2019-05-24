from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

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

# The fraction of the response variance that is captured by the model.
print("R^2 train: {:.3f}.".format(r2_score(y_true=y_train,
                                           y_pred=y_train_pred)))
print("R^2 test: {:.3f}.".format(r2_score(y_true=y_test,
                                          y_pred=y_test_pred)))
