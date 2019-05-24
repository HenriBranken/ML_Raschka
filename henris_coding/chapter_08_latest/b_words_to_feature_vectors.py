import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
    "The sun is shining",
    "The weather is sweet",
    "The sun is shining, the weather is sweet, and one and one is two"
])

bag = count.fit_transform(docs)

# Print the contents of the vocabulary.
# The vocabulary maps the unique words to integer indices.
print(count.vocabulary_)

# Print the feature vectors that we have created:
print(bag.toarray())
