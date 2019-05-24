import pyprind
import pandas as pd
import numpy as np
import os
import re


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


basepath = "/home/henri/stuff/machine_learning/sebastian_raschka/" \
           "henris_coding/chapter_08_latest/aclImdb"

labels = {"pos": 1, "neg": 0}
pbar = pyprind.ProgBar(iterations=50000)

df = pd.DataFrame()
for s in ("test", "train"):
    for l in ("pos", "neg"):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                txt = f.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()

df.columns = ["review", "sentiment"]

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

df["review"] = df["review"].apply(preprocessor)

df.to_csv("movie_data.csv", index=False, encoding="utf-8")

df = pd.read_csv("movie_data.csv", encoding="utf-8")
print(df.head(3))
print(df.tail(3))