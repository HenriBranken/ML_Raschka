import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

df = pd.read_csv("./wdbc.data", header=None)

X = df.loc[:, 2:].values  # features
y = df.loc[:, 1].values  # labels
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    stratify=y,
                                                    random_state=1)

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

# Now use the matshow function to make the results a little bit easier to
# interpret.
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.7)
for i in range(confmat.shape[0]):  # loop over the rows (y)
    for j in range(confmat.shape[1]):  # loop over the columns (x)
        ax.text(x=j, y=i, s=confmat[i, j], va="center", ha="center")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
