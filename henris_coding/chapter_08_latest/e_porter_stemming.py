from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


print(tokenizer_porter("Runners like running and thus they run"))
