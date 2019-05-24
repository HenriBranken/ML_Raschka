from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_wine = pd.read_csv("./wine.data", header=None)
df_wine.columns = ["class_label", "alcohol", "malic_acid", "ash",
                   "alcalinity_of_ash", "magnesium", "total_phenols",
                   "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                   "color_intensity", "hue", "od280/od315_of_diluted_wines",
                   "proline"]

# Drop 1 class
df_wine = df_wine[df_wine["class_label"] != 1]
y = df_wine["class_label"].values
X = df_wine[["alcohol", "od280/od315_of_diluted_wines"]].values

# Encode the class labels into binary format.
# Split the dataset into 80 percent training and 20 percent test sets.
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1, stratify=y)

tree = DecisionTreeClassifier(criterion="entropy", random_state=1, max_depth=1)
ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)
# n_estimators is the maximum number of estimators at which boosting is
# terminated.  In case of perfect fit, the learning procedure is stopped early.
# learning_rate shrinks the contribution of each classifier by learning_rate.
# There is a trade-off between learning_rate and n_estimators.
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train_score = accuracy_score(y_true=y_train, y_pred=y_train_pred)
tree_test_score = accuracy_score(y_true=y_test, y_pred=y_test_pred)
print("Decision Tree train/test accuracies: "
      "{:.3f}/{:.3f}.".format(tree_train_score, tree_test_score))

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train_score = accuracy_score(y_true=y_train, y_pred=y_train_pred)
ada_test_score = accuracy_score(y_true=y_test, y_pred=y_test_pred)
print("AdaBoost train/test accuracies:  "
      "{:.3f}/{:.3f}.".format(ada_train_score, ada_test_score))

# What do the decision regions look like:
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axs = plt.subplots(nrows=1, ncols=2, sharex="col", sharey="row",
                      figsize=(12, 5))
for idx, clf, tt in zip([0, 1], [tree, ada], ["Decision Tree", "AdaBoost"]):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axs[idx].contourf(xx, yy, Z, alpha=0.3)
    axs[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                     c="blue", marker="^")
    axs[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                     c="red", marker="o")
    axs[idx].set_title(tt)
    axs[idx].set_xlabel("OD280/OD315 of diluted wines")
axs[0].set_ylabel("Alcohol content")
plt.tight_layout()
plt.show()
