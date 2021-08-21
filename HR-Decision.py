print("-------------- start --------------")

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.metrics import classification_report
import os

for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# read the input data
df = pd.read_excel('./input/HR_Employee_Data.xlsx')

# get features names
features = df[
    ['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary', 'Work_accident', 'Department']]

# convert the string data into numerical data so that can be used as a features
le_salary = LabelEncoder()
features['salary_label'] = le_salary.fit_transform(features['salary'])

features = features.drop(['salary', 'Department'], axis='columns')

# normalize the data using MinMaxScaler
features_scaler = MinMaxScaler()
features = features_scaler.fit_transform(features)

x = features
y = df.left

# list for saving models scores
scores = []

# set the models parameters
model_params = {
    'svm': {
        'model': SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20, 30],
            'kernel': ['rbf', 'linear', 'poly']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1, 5, 10]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 7, 11, 13]
        }
    }

}

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df_score = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df_score)

# # split the data set into train and test
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
#
# model = RandomForestClassifier(n_estimators=100)
# model.fit(x_train, y_train)
# print(model.score(x_test, y_test))
#
# y_predicted = model.predict(x_test)
# cm = confusion_matrix(y_test, y_predicted)
# plt.figure(figsize=(10, 7))
# sb.heatmap(cm, annot=True, fmt=".1f")
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
#
# # using Random Forest to create Confusion Matrix and Classification Report
# print(classification_report(y_test, y_predicted))

print("-------------- end --------------")
