# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:56:50 2018

@author: 백인혁
"""

from __future__ import print_function
import os
# Type Path
data_path = ['C:\\pratice\\data']

def accuracy(real,predict):
    return sum(real == predict)/float(real.shape[0])

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

#Wine_DATA
filepath = os.sep.join(data_path + ['Wine_Quality_Data.csv'])
data = pd.read_csv(filepath, sep=',')
X_data = data.copy()
y_data = X_data.pop('color')
result = pd.Series()

from sklearn.preprocessing import StandardScaler
StdSc = StandardScaler()

StdSc = StdSc.fit(X_data)
X_scaled = StdSc.transform(X_data)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.3)
SVM = SVC(kernel='rbf', degree= 9 , C = 6.0)
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)

print('Wine Quality best case : ',accuracy(y_test, y_pred))


#Testing 10 times
print("---Testing 10 times---")

for k in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.3)
    SVM = SVC(kernel='rbf', degree= 9 , C = 6.0)
    SVM.fit(X_train, y_train)
    y_pred = SVM.predict(X_test)
    result.at[k] = accuracy(y_test,y_pred)
    print('Wine Result  :  ',accuracy(y_test,y_pred))

print('10times mean : ',result.mean())












