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
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd


import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

#IRIS_DATA
filepath = os.sep.join(data_path + ['Iris_Data.csv'])
data = pd.read_csv(filepath)
X_data = data.copy()
y_data = X_data.pop('species')
result = pd.Series()

mmc = MinMaxScaler()
X_data = mmc.fit_transform(X_data)
X_data = mmc.transform(X_data)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
LDA = LinearDiscriminantAnalysis(n_components=2)
LDA.fit(X_train, y_train)
y_pred = LDA.predict(X_test)

print('IRIS best case : ',accuracy(y_test, y_pred))


#Testing 10 times
print("---Testing 10 times---")

X_data = data.copy()
y_data = X_data.pop('species')
result = pd.Series()

for k in range(10):
    mmc = MinMaxScaler()
    X_data = mmc.fit_transform(X_data)
    X_data = mmc.transform(X_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    LDA = LinearDiscriminantAnalysis(n_components=2)
    LDA.fit(X_train, y_train)
    y_pred = LDA.predict(X_test)
    result.at[k] = accuracy(y_test,y_pred)

print(result)
print('10times mean : ',result.mean())














