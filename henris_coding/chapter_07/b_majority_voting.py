from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from MajorityVoteClassifier import MajorityVoteClassifier

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

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring="roc_auc")
    print("ROC_AUC: {:.3f} (+/- {:.3f}) [{:s}]".format(scores.mean(),
                                                       scores.std(),
                                                       label))
