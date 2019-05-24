import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

df = pd.read_csv("./wdbc.data", header=None)

X = df.loc[:, 2:].values  # features
y = df.loc[:, 1].values  # labels
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)

print("Precision: {:.3f}.".format(precision_score(y_true=y_test,
                                                  y_pred=y_pred)))
print("Recall: {:.3f}.".format(recall_score(y_true=y_test,
                                            y_pred=y_pred)))
print("F1 score: {:.3f}.".format(f1_score(y_true=y_test,
                                          y_pred=y_pred)))
