from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

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

fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(start=-4, stop=6+1, step=1):
    lr = LogisticRegression(penalty="l1", C=10.0**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.0**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column + 1],
             color=color)
plt.axhline(y=0, color="black", linestyle="--", linewidth=3)
plt.xlim([10**(-4), 10**6])
plt.ylabel("Weight Coefficient")
plt.xlabel("C")
plt.xscale("log")
plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, fancybox=True)
plt.show()
