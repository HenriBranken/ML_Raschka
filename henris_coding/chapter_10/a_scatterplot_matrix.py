import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./housing_data.txt", sep="\s+", header=None)

df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
              "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

cols = ["LSTAT", "INDUS", "NOX", "RM", "MEDV"]

sns.pairplot(df[cols], height=2.5)
plt.tight_layout()
plt.show()
