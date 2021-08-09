# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('./input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_excel('./input/HR_Employee_Data.xlsx')
df.head()
df.info()

df.groupby('left').mean()

pd.crosstab(df.number_project, df.left).plot(kind='bar')

pd.crosstab(df.salary, df.left).plot(kind='bar')

pd.crosstab(df.Work_accident, df.left).plot(kind='bar')

pd.crosstab(df.Department, df.left).plot(kind='bar')

pd.crosstab(df.time_spend_company, df.left).plot(kind='bar')

pd.crosstab(df.promotion_last_5years, df.left).plot(kind='bar')

plt.show()
