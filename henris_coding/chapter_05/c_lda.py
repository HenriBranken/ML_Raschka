from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt


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

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
plot_decision_regions(X=X_train_lda, y=y_train, classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="best")
plt.pause(5)
plt.close()

# Let us take a look at the results on the test set:
X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel("LD 1")
plt.ylabel("LD 2")
plt.legend(loc="best")
plt.pause(5)
plt.close()
# As we see, the logistic regression classifier is able to get a perfect
# accuracy score for classifying the samples in the test dataset by only using
# a 2-dimensional feature subspace instead of the original 13 Wine features.
