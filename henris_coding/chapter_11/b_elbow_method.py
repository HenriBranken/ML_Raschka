from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)

# The idea behind the elbow method is to identify the value of k where the
# distortion begins to increase most rapidly.
plt.plot(range(1, 11), distortions, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.grid()
plt.tight_layout()
plt.show()
