import numpy as np
import matplotlib.pyplot as plt
# roc curve and auc score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


# Define a Python Function to plot the ROC curves:
def plot_roc_curve(fprates, tprates):
    plt.plot(fprates, tprates, color="orange", label="ROC")
    plt.plot([0, 1], [0, 1], color="darkblue", linestyle="--")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.show()


# Generate the sample data
X_data, y = make_classification(n_samples=1000, n_classes=2,
                                weights=[1, 1], random_state=1)
print("data_X.shape = {}.".format(X_data.shape))  # 1000 samples, 20 features
print("class_label.shape = {}.".format(y.shape))  # 1000 labels
print("sum(y == 0) = {}.".format(sum(y == 0)))  # 499 class-0 labels
print("sum(y == 1) = {}.".format(sum(y == 1)))  # 501 class-1 labels

# Split the data into train and test sub-datasets
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.3,
                                                    random_state=1, stratify=y)

# Fit a model on the TRAIN data
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict the probabilities for the test data
probs = model.predict_proba(X_test)  # probabilities for the targets 0 & 1.
print("probs.shape = {}.".format(probs.shape))  # (300, 2)

# Keep the probabilities of the positive class only
probs = probs[:, 1]
print("probs.shape = {}.".format(probs.shape))  # (300, )

# Compute the AUC score
auc = roc_auc_score(y_true=y_test, y_score=probs)
print("AUC: {:.2f}.".format(auc))

# Get the ROC Curve
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=probs)
print("len(fpr), len(tpr), len(thresholds): "
      "{}, {}, {}.".format(len(fpr), len(tpr), len(thresholds)))
print("min(fpr), max(fpr): {}, {}.".format(min(fpr), max(fpr)))
print("min(tpr), max(tpr): {}, {}.".format(min(tpr), max(tpr)))
print("min(thresholds), max(thresholds): {}, {}.".format(min(thresholds),
                                                         max(thresholds)))
print("np.mean(fpr), np.mean(tpr), np.mean(thresholds): "
      "{:.3f}, {:.3f}, {:.3f}.".format(np.mean(fpr), np.mean(tpr),
                                       np.mean(thresholds)))
print("thresholds = {}.".format(thresholds.reshape(-1, 1)))

plot_roc_curve(fprates=fpr, tprates=tpr)
