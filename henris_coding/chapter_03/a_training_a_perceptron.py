from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # The petal length and the petal width
y = iris.target
print("The class labels are as follows:  {}.".format(np.unique(y)))

# Split the dataset into separate training and test datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1,
                                                    stratify=y)

# Standardize the features
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train a perceptron model
ppn = Perceptron(max_iter=50, eta0=0.1, random_state=1, n_jobs=-1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print("The number of misclassified samples is:"
      "  {}.".format(np.sum((y_test != y_pred))))
print("The accuracy score is:  {:.3f}.".format(accuracy_score(y_test, y_pred)))
print("The accuracy score is:  {:.3f}.".format(ppn.score(X_test_std, y_test)))


X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
a = np.where(y_combined == y_test)
print(a)
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
                      test_idx=range(105, 150))
plt.xlabel("petal length [standardised]")
plt.ylabel("petal width [standardised]")
plt.legend()
plt.tight_layout()
plt.show()
