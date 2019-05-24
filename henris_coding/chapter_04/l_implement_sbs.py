from k_sbs import SBS
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv("./wine_data.txt", header=None)
df_wine.columns = ["class_label", "alcohol", "malic_acid", "ash",
                   "alcalinity_of_ash", "magnesium", "total_phenols",
                   "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                   "color_intensity", "hue", "od280/od315_of_diluted_wines",
                   "proline"]

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5)

sbs = SBS(estimator=knn, k_features=1)
sbs.fit(X_train_std, y_train)

k_feat = [len(k) for k in sbs.subsets_]

# Plot the classification accuracy of the KNN classifier that was calculated
# on the validation dataset:
plt.plot(k_feat, sbs.scores_, marker="o")
plt.ylim([0.7, 1.02])
plt.ylabel("Accuracy")
plt.xlabel("Number of Features")
plt.grid()
plt.pause(interval=2)

# What is the smallest features subset (k=3) that yielded a good performance
# on the validation dataset (accuracy = 1.00):
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

# Let us evaluate the performance of the KNN classifier on the original test
# set.  Here we use the complete feature set.
knn.fit(X_train, y_train)
print("Training Accuracy = {:.4f}.".format(knn.score(X_train_std, y_train)))
print("Test Accuracy = {:.4f}.".format(knn.score(X_test_std, y_test)))

# Let us use the selected three-feature subset and see how well KNN performs:
knn.fit(X_train_std[:, k3], y_train)
print("\nTraining Accuracy = {:.4f}.".format(knn.score(X_train_std[:, k3],
                                                       y_train)))
print("Test Accuracy = {:.4f}.".format(knn.score(X_test_std[:, k3],
                                                 y_test)))
