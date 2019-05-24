from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


porter = PorterStemmer()
stop = stopwords.words("english")


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def apply_stops(text):
    stems = tokenizer_porter(text)
    return [word for word in stems if word not in stop]


sentence = "a runner likes running and runs a lot"
print(apply_stops(sentence))
