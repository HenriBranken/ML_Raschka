import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, \
    cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("./wdbc.data", header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)

gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{"max_depth": [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring="accuracy",
                  cv=2,
                  n_jobs=-1)

scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)

print("CV accuracy: {:.3f} +/- {:.3f}.".format(np.mean(scores),
                                               np.std(scores)))
