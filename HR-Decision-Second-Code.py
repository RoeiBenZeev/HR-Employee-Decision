import os
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

print()
print("-------------- read the data and analyze the columns --------------")
np.random.seed(42)
for dir_name, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dir_name, filename))

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)
df = pd.read_excel("./input/HR_Employee_Data.xlsx")
df.head()
df.drop(['Emp_Id'], axis=1, inplace=True)
df.info()
df.corr().style.background_gradient(sns.light_palette('green', as_cmap=True))
list(enumerate(df.drop(['left'], 1).columns))

ct = ColumnTransformer([
    ("one-hot", OneHotEncoder(), [7]),
    ("ordinal", OrdinalEncoder(), [8])
], remainder='passthrough')

print()
print("-------------- run all models - start --------------")

pipes = {}
models = {}

# set all models
models['SVC'] = SVC(probability=True)
models['LR'] = LogisticRegression()
models['DT'] = DecisionTreeClassifier()
models['RF'] = RandomForestClassifier()
models['XGB'] = XGBClassifier(use_label_encoder=False)

# iterate over all models
for m in models:
    pipes[m] = Pipeline([
        ("columns compose", ct),
        ("standardize", StandardScaler()),
        (m, models[m])
    ])

x, y = df.drop(['left'], axis=1), df['left']
le = LabelEncoder().fit(y)
y = le.transform(y)

# separate the data set to train set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=112)

for p in pipes:
    pipes[p].fit(x_train, y_train)

print("-------------- run all models - finish --------------")
print()
print("-------------- print models scores --------------")

for p in pipes:
    print(f'\t\t{p}')
    args = dict(y_true=y_test, y_pred=pipes[p].predict(x_test))
    print(confusion_matrix(**args))
    print(classification_report(**args))
    print("=" * 60)

# print the graphs
print("-------------- print graph of all models --------------")
print()

plt.figure(figsize=(13, 6))

for p in pipes:
    y_pred = pipes[p].predict_proba(x_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1].ravel())
    plt.plot(fpr, tpr, label=p)

plt.title('ROC curve')
plt.xlabel('False-Positive rate')
plt.ylabel('True-Positive rate')
plt.legend()
plt.show()
