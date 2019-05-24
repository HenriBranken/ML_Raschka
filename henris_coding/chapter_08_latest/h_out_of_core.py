import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import pyprind
stop = stopwords.words("english")


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # skip the header, which is the first line.
        for line in f:
            inf_list = line.split(",")
            text = inf_list[0].lstrip().rstrip()
            label = int(inf_list[1].lstrip().rstrip())
            yield text, label


def get_minibatch(size, doc_stream=stream_docs("movie_data.csv")):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


vect = HashingVectorizer(decode_error="ignore",
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss="log", random_state=1, max_iter=1, tol=1e-3)

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])

for _ in range(45):
    X_train, y_train = get_minibatch(size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)  # transform a sequence of docs to a
    # document-term matrix
    clf.partial_fit(X_train, y_train, classes=classes)  # Perform one epoch of
    # stochastic gradient descent on the given samples.
    pbar.update()

# Use the last 5000 documents to evaluate the performance of our model.
X_test, y_test = get_minibatch(size=5000)
X_test = vect.transform(X_test)
print("Accuracy: {:.3f}.".format(clf.score(X_test, y_test)))

# Finally, we can use the last 5000 document to update our model
clf = clf.partial_fit(X_test, y_test)
