from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Train an SVM model to classify the different flower in our Iris dataset:
svm = SVC(kernel="linear", C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel("Petal Length [standardized]")
plt.ylabel("Petal Width [standardized]")
plt.legend(loc="upper left")
plt.show()
