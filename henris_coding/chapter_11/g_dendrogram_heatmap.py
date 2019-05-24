from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
variables = ["X", "Y", "Z"]
labels = ["ID_0", "ID_1", "ID_2", "ID_3", "ID_4"]
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)

row_clusters = linkage(df.values,
                       method="complete",
                       metric="euclidean")

# 1.
# Create a new figure object.
# Define the x axis position, y axis position, width and height of the dendr
# via the add_axes attribute.
# Rotate the dendrogram 90 degrees counter-clockwise.
fig = plt.figure(figsize=(8, 8), facecolor="white")
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation="left")

# 2.
# Reorder the data in our initial df according to the clustering labels that
# can be accessed from the dendrogram object (which is essentially a Python
# dictionary) via the `leaves` key.
df_rowclust = df.iloc[row_dendr["leaves"][::-1]]

# 3.
# Construct the heat map from the reordered df and position it next to the
# dendrogram.
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation="nearest", cmap="hot_r")

# 4.
# Modify the aesthetics of the dendrogram by removing the axis ticks and hiding
# the axis spines.
# Also add a color bar and assign the features and sample names to the x and y
# axis tick labels, respectively.
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([""] + list(df_rowclust.columns))
axm.set_yticklabels([""] + list(df_rowclust.index))
plt.show()
