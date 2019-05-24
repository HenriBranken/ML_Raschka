import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./housing_data.txt", sep="\s+", header=None)

df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

cols = ["LSTAT", "INDUS", "NOX", "RM", "MEDV"]

corr_matrix = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)

hm = sns.heatmap(data=corr_matrix,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt=".2f",
                 annot_kws={"size": 15},
                 yticklabels=cols,
                 xticklabels=cols)
plt.tight_layout()
plt.show()
