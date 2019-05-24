import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv("movie_data.csv", encoding="utf-8")

# Use CountVectorizer to create the bag-of-words matrix as input to the LDA.
count = CountVectorizer(stop_words="english",
                        max_df=0.1,
                        max_features=5000)
X = count.fit_transform(df["review"].values)

lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method="batch",
                                n_jobs=-1)
# Let the lda estimator do its estimation based on all the availabel training
# data (bag-of-words matrix) in one iteration.

X_topics = lda.fit_transform(X)  # fit to the data, then transform it.

components = lda.components_
print(components[0, 0: 10])

n_top_words = 5
feature_names = count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic {:.0f}:".format(topic_idx + 1))
    print(" ".join([feature_names[i] for i in
                    topic.argsort()[-1: -n_top_words - 1: -1]]))
    print("\n")

music = X_topics[:, 7].argsort()[::-1]
for iter_idx, movie_idx in enumerate(music[:3]):
    print("\nMusic Movie {:.0f}:".format(iter_idx + 1))
    print(df["review"][movie_idx][:300], "...")
