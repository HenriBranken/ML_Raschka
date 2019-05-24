from scipy.cluster.hierarchy import linkage
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)

np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)

row_clusters = linkage(df.values,
                       method="complete",
                       metric="euclidean")

linkage_matrix = pd.DataFrame(row_clusters,
                              columns=["row label 1", "row label 2",
                                       "distance",
                                       "no. of items in the cluster"],
                              index=["cluster {:.0f}.".format(i + 1) for i in
                                     range(row_clusters.shape[0])])

print(linkage_matrix)
