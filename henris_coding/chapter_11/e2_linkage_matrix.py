from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)

np.random.seed(124)
variables = ["X", "Y"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = np.random.randint(low=0, high=10, size=(5, 2))
df = pd.DataFrame(X, columns=variables, index=labels)
print(df)

row_dist = pd.DataFrame(data=squareform(pdist(df.values, metric="euclidean")),
                        columns=labels, index=labels)
print(row_dist)

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
