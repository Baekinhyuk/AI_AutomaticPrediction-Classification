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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier


import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

#Orange_DATA
filepath = os.sep.join(data_path + ['Orange_Telecom_Churn_Data.csv'])
data = pd.read_csv(filepath)
data.drop(['state', 'area_code', 'phone_number'], axis=1, inplace=True)

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

for col in ['intl_plan', 'voice_mail_plan']:
    data[col] = lb.fit_transform(data[col])

X_data = data.copy()
y_data = X_data.pop('churned')
result = pd.Series()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
clf1 = BaggingClassifier(n_estimators=400)
clf2 = ExtraTreesClassifier(n_estimators=400, max_features=9)
clf3 = GradientBoostingClassifier(learning_rate=0.1, max_features=9, subsample=0.5, n_estimators=400)
eclf = VotingClassifier(estimators=[('bc', clf1), ('rc', clf2), ('ec', clf3)],voting='hard')
eclf = eclf.fit(X_train, y_train)
y_pred = eclf.predict(X_test)

print('ORANGE TEL best case : ',accuracy(y_test, y_pred))


#Testing 10 times
print("---Testing 10 times---")

for k in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    clf1 = BaggingClassifier(n_estimators=400)
    clf2 = ExtraTreesClassifier(n_estimators=400, max_features=9)
    clf3 = GradientBoostingClassifier(learning_rate=0.1, max_features=9, subsample=0.5, n_estimators=400)
    eclf = VotingClassifier(estimators=[('bc', clf1), ('rc', clf2), ('ec', clf3)],voting='hard')
    eclf = eclf.fit(X_train, y_train)
    y_pred = eclf.predict(X_test)
    result.at[k] = accuracy(y_test,y_pred)
    print('Voting Result  :  ',accuracy(y_test,y_pred))

print('10times mean : ',result.mean())












