from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import set_link_color_palette
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

set_link_color_palette(["black"])
pd.set_option('display.max_columns', 500)

np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)

row_clusters = linkage(df.values,
                       method="complete",
                       metric="euclidean")

row_dendr = dendrogram(row_clusters,
                       labels=np.asarray(labels),
                       color_threshold=np.inf)


plt.tight_layout()
plt.ylabel("Euclidean Distance")
plt.show()
