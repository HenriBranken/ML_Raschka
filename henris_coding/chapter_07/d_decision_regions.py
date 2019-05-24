from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from MajorityVoteClassifier import MajorityVoteClassifier
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=1, stratify=y)

clf_1 = LogisticRegression(penalty="l2", C=0.001, random_state=1,
                           solver="lbfgs")
clf_2 = DecisionTreeClassifier(max_depth=1, criterion="entropy",
                               random_state=0)
clf_3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")

pipe_1 = Pipeline([("sc", StandardScaler()),
                   ("clf", clf_1)])
pipe_3 = Pipeline([("sc", StandardScaler()),
                   ("clf", clf_3)])

mv_clf = MajorityVoteClassifier(classifiers=[pipe_1, clf_2, pipe_3])
clf_labels = ["Logistic_Regression", "Decision_Tree", "KNN", "Majority_Voting"]
all_clf = [pipe_1, clf_2, pipe_3, mv_clf]

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axs = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row",
                      figsize=(8, 5))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axs[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axs[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0],
                                X_train_std[y_train == 0, 1], c="blue",
                                marker="^", s=50)
    axs[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                X_train_std[y_train == 1, 1], c="green",
                                marker="o", s=50)
    axs[idx[0], idx[1]].set_title(tt)
plt.tight_layout()
plt.show()
