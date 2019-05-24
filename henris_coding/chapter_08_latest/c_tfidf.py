import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
np.set_printoptions(precision=2)

count = CountVectorizer()
docs = np.array([
    "The sun is shining",
    "The weather is sweet",
    "The sun is shining, the weather is sweet, and one and one is two"
])

bag = count.fit_transform(docs)

tfidf = TfidfTransformer(use_idf=True, norm="l2", smooth_idf=True)
# Transform a count matrix to a normalised tf or tf-df representation.
# Enable inverse-document-frequency re-weighting.
# Prevents zero division.

tfidf = tfidf.fit_transform(bag)

print(tfidf.toarray())
