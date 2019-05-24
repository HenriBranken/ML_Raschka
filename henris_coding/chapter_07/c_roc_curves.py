from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from MajorityVoteClassifier import MajorityVoteClassifier
import matplotlib.pyplot as plt

colors = ["black", "orange", "blue", "green"]
linestyles = [":", "--", "-.", "-"]

iris = datasets.load_iris()
X, y = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                    random_state=1, stratify=y)

clf_1 = LogisticRegression(penalty="l2", C=0.001, random_state=1,
                           solver="lbfgs")
clf_2 = DecisionTreeClassifier(max_depth=1, criterion="entropy",
                               random_state=0)
clf_3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")

pipe_1 = Pipeline([("sc", StandardScaler()),
                   ("clf", clf_1)])
pipe_3 = Pipeline([("sc", StandardScaler()),
                   ("clf", clf_3)])

mv_clf = MajorityVoteClassifier(classifiers=[pipe_1, clf_2, pipe_3])
clf_labels = ["Logistic_Regression", "Decision_Tree", "KNN", "Majority_Voting"]
all_clf = [pipe_1, clf_2, pipe_3, mv_clf]

for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # assuming the label of the positive class is 1.
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, ls=ls,
             label="{:s} (auc = {:.3f})".format(label, roc_auc))
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid(alpha=0.4)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve for four different classifiers")
plt.tight_layout()
plt.show()
