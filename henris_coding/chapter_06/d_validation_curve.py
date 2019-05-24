import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

df = pd.read_csv("./wdbc.data", header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)

pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty="l2",
                                           random_state=1,
                                           solver="lbfgs",
                                           max_iter=1000))

train_scores, test_scores = \
    validation_curve(estimator=pipe_lr, X=X_train, y=y_train,
                     param_name="logisticregression__C",
                     param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color="blue", marker="o", markersize=5,
         label="Training Accuracy")
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std,
                 alpha=0.15, color="blue")
plt.plot(param_range, test_mean, color="green", linestyle="--", marker="s",
         markersize=5, label="Validation Accuracy")
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std,
                 alpha=0.15, color="green")
plt.grid()
plt.xscale("log")
plt.legend(loc="lower right")
plt.xlabel("Inverse Regularization Strength, C")
plt.ylabel("Accuracy")
plt.show()
