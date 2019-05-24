# By executing the following code, we will now implement a KNN model in scikit
# learn using a Euclidean distance metric.


from sklearn.neighbors import KNeighborsClassifier
from plot_decision_regions import plot_decision_regions
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # The petal length and the petal width
y = iris.target

# Split the dataset into separate training and test datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1,
                                                    stratify=y)
y_combined = np.hstack((y_train, y_test))
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski")
# Minkowski is a generalisation of the Euclidean and Manhattan distance.
# It becomes the Euclidean distance if we set the parameter p=2, or the
# Manhattan distance at p=1.
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=knn,
                      test_idx=range(105, 150))
plt.xlabel("Petal Length [standardised]")
plt.ylabel("Petal Width [standardised]")
plt.legend(loc="upper left")
plt.show()
