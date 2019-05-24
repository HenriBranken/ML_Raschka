from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)

# Clustering data of arbitrary shapes.
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric="euclidean")
y_db = db.fit_predict(X)

plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1],
            c="lightblue", edgecolor="black",
            marker="o", s=40, label="Cluster 1")
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
            c="red", edgecolor="black",
            marker="s", s=40, label="Cluster 2")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("DBSCAN algorithm")
plt.tight_layout()
plt.show()
