import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv("./wdbc.data", header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)

X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

print("Number of class 1 samples before: "
      "{:.0f}.".format(X_imb[y_imb == 1].shape[0]))

X_upsampled, y_upsampled = resample(X_imb[y_imb == 1],
                                    y_imb[y_imb == 1],
                                    replace=True,
                                    n_samples=X_imb[y_imb == 0].shape[0],
                                    random_state=123)

print("Number of class 1 samples after: "
      "{:.0f}.".format(X_upsampled.shape[0]))

# After resampling, we can then stack the original class 0 samples with the
# upsampled class 1 subset to obtain a balanced dataset as follows:
X_bal = np.vstack((X[y == 0], X_upsampled))
y_bal = np.hstack((y[y == 0], y_upsampled))

# Consequently a majority vote prediction rule would only achieve 50% accuracy.
y_pred = np.zeros(y_bal.shape[0])
print(np.mean(y_pred == y_bal) * 100)
