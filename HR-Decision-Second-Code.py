import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)
for dir_name, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dir_name, filename))

pd.set_option('mode.chained_assignment', None)
df = pd.read_excel("./input/HR_Employee_Data.xlsx")
df.head()
df.drop(['Emp_Id'], axis=1, inplace=True)
df.info()
df.corr().style.background_gradient(sns.light_palette('green', as_cmap=True))

list(enumerate(df.drop(['left'], 1).columns))

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

ct = ColumnTransformer([
    ("one-hot", OneHotEncoder(), [7]),
    ("ordinal", OrdinalEncoder(), [8])
], remainder='passthrough')

pipes = {}
models = {}

models['SVC'] = SVC(probability=True)
models['LR'] = LogisticRegression()
models['DT'] = DecisionTreeClassifier()
models['RF'] = RandomForestClassifier()
models['XGB'] = XGBClassifier(use_label_encoder=False)

for m in models:
    pipes[m] = Pipeline([
        ("columns compose", ct),
        ("standardize", StandardScaler()),
        (m, models[m])
    ])

X, y = df.drop(['left'], axis=1), df['left']

le = LabelEncoder().fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=112)

for p in pipes:
    pipes[p].fit(X_train, y_train)

for p in pipes:
    print(f'\t\t{p}')
    args = dict(y_true=y_test, y_pred=pipes[p].predict(X_test))
    print(confusion_matrix(**args))
    print(classification_report(**args))
    print("=" * 60)
