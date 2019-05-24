import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# Use unpruned decision tree as the base classifier, and create an ensemble of
# 500 decision trees fit on different bootstrap samples of the training dataset
tree = DecisionTreeClassifier(criterion="entropy", random_state=1,
                              max_depth=None)
# If `None`, then the nodes are expanded until all the leaves are pure or until
# all the leaves contain less than `min_samples_split` samples.
bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs=-1,
                        random_state=1)

# Calculate the accuracy score of the prediction on the training set and test
# dataset to compare the performance of the bagging classifier to the
# performance of a single unpruned decision tree.
tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train_score = accuracy_score(y_true=y_train, y_pred=y_train_pred)
tree_test_score = accuracy_score(y_true=y_test, y_pred=y_test_pred)
print("Decision Tree train/test accuracies: "
      "{:.3f}/{:.3f}.".format(tree_train_score, tree_test_score))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train_score = accuracy_score(y_true=y_train, y_pred=y_train_pred)
bag_test_score = accuracy_score(y_true=y_test, y_pred=y_test_pred)
print("Bagging train/test accuracies: "
      "{:.3f}/{:.3f}.".format(bag_train_score, bag_test_score))

# Compare the decision regions between the decision tree and the bagging
# classifier:
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows=1, ncols=2, sharex="col", sharey="row",
                        figsize=(10, 5))
for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision Tree', 'Bagging']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                       c="blue", marker="^")
    axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                       c="green", marker="o")
    axarr[idx].set_title(tt)
    axarr[idx].set_xlabel("OD280/OD315 of diluted wines", fontsize=12)
axarr[0].set_ylabel("Alcohol", fontsize=12)
plt.tight_layout()
plt.show()
