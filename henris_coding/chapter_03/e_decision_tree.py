from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plot_decision_regions import plot_decision_regions
import numpy as np
import matplotlib.pyplot as plt
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # The petal length and the petal width
y = iris.target

# Split the dataset into separate training and test datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1, stratify=y)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

tree = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=1)
tree.fit(X_train, y_train)

# plot_decision_regions(X_combined, y_combined, classifier=tree,
# test_idx=range(105, 150))
# plt.xlabel("Petal Length [cm]")
# plt.ylabel("Petal Width [cm]")
# plt.legend(loc="upper left")
# plt.show()

dot_data = export_graphviz(tree, filled=True, rounded=True,
                           class_names=["Setosa", "Versicolor", "Virginica"],
                           feature_names=["petal length", "petal_width"],
                           out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_png("tree.png")
