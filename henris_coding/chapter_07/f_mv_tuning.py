from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from MajorityVoteClassifier import MajorityVoteClassifier
import pandas as pd
pd.set_option('display.max_columns', 500)

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

params = {"decisiontreeclassifier__max_depth": [1],
          "pipeline-1__clf__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0,
                                 1000.0]}
grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring="roc_auc",
                    iid=False,
                    return_train_score=True)
grid.fit(X_train, y_train)

print("Best Parameters: {}.\n".format(grid.best_params_))

print("Accuracy: {:.3f}.\n".format(grid.best_score_))

output = grid.cv_results_.copy()
my_dict = {"mean_test_score": output["mean_test_score"],
           "std_test_score": output["std_test_score"],
           "max_depth": output["param_decisiontreeclassifier__max_depth"],
           "C": output["param_pipeline-1__clf__C"]}
df = pd.DataFrame(data=my_dict)
print(df)
