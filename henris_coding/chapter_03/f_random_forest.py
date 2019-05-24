from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from plot_decision_regions import plot_decision_regions
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # The petal length and the petal width
y = iris.target

# Split the dataset into separate training and test datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1, stratify=y)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

forest = RandomForestClassifier(criterion="gini",
                                n_estimators=25,
                                random_state=1,
                                n_jobs=-1)

forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest,
                      test_idx=range(105, 150))
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend(loc="upper left")
plt.show()
