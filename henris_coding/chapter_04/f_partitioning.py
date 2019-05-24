import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
# `None` forces to display any number of columns.


df_wine = pd.read_csv("./wine_data.txt", header=None)
df_wine.columns = ["class_label", "alcohol", "malic_acid", "ash",
                   "alcalinity_of_ash", "magnesium", "total_phenols",
                   "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                   "color_intensity", "hue", "od280/od315_of_diluted_wines",
                   "proline"]
print("Class labels: {}.".format(np.unique(df_wine["class_label"])))

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)
