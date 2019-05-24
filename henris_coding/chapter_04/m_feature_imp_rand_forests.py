import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500, random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("{:>2}) {:<30} {:.4f}".format(f+1, feat_labels[indices[f]],
                                        importances[indices[f]]))

# Create a plot that ranks the different features in the Wine dataset by their
# relative importance; note that the feature importance values are normalised
# so that they sum up to 1.0:
plt.title("Feature Importance")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")

plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.pause(10)
