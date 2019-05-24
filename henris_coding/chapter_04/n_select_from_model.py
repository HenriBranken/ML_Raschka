from sklearn.feature_selection import SelectFromModel
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

feat_labels = df_wine.columns[1:]
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)
forest = RandomForestClassifier(n_estimators=500, random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

# SelectFromModel select features based on a user-specified threshold after
# model fitting, which is useful if we want to use the RFC as a feature
# selector and intermediate step in a scikit-learn Pipeline object, which
# allows us to connect different preprocessing steps with an estimator.
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X)

for f in range(X_selected.shape[1]):
    print("{:>2}) {:<30} {:.4f}".format(f + 1, feat_labels[indices[f]],
                                        importances[indices[f]]))
