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
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier


import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

#Smartphone_DATA
filepath = os.sep.join(data_path + ['Human_Resources_Employee_Attrition.csv'])
data = pd.read_csv(filepath)

X_data = data.copy()
y_data = X_data.pop('salary')
z_data = X_data.pop('department')

result = pd.Series()

le = LabelEncoder()
le.fit(z_data)
le.classes_
department_le = le.transform(z_data)
X_data['department'] = department_le

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
LR = LogisticRegression(penalty='l2', C=10.0)
clf1 = ExtraTreesClassifier(n_estimators=100, max_features=4)
clf2 = LogisticRegressionCV(random_state=0, multi_class='multinomial', cv=6)
clf3 = GaussianNB()
clf4 = GridSearchCV(LR, param_grid={'C':[0.001, 0.01, 0.1]}, scoring='accuracy', cv=6)
eclf = VotingClassifier(estimators=[('ec', clf1), ('abc', clf2), ('rf', clf3), ('bc', clf4)],voting='soft',weights=[2,1,1,1])
eclf = eclf.fit(X_train, y_train)
y_pred = eclf.predict(X_test)

print('Employee best case : ',accuracy(y_test, y_pred))


#Testing 10 times
print("---Testing 10 times---")

for k in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    LR = LogisticRegression(penalty='l2', C=10.0)
    clf1 = ExtraTreesClassifier(n_estimators=100, max_features=4)
    clf2 = LogisticRegressionCV(random_state=0, multi_class='multinomial', cv=6)
    clf3 = GaussianNB()
    clf4 = GridSearchCV(LR, param_grid={'C':[0.001, 0.01, 0.1]}, scoring='accuracy', cv=6)
    eclf = VotingClassifier(estimators=[('ec', clf1), ('abc', clf2), ('rf', clf3), ('bc', clf4)],voting='soft',weights=[2,1,1,1])
    eclf = eclf.fit(X_train, y_train)
    y_pred = eclf.predict(X_test)
    result.at[k] = accuracy(y_test,y_pred)
    print('Voting Result  :  ',accuracy(y_test,y_pred))

print('10times mean : ',result.mean())












