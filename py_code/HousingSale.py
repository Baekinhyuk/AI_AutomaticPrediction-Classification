# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:56:50 2018

@author: 백인혁
"""

from __future__ import print_function
import os
# Type Path
data_path = ['C:\\pratice\\data']

def rmse(ytrue, ypredicted):
    return np.sqrt(mean_squared_error(ytrue, ypredicted))

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor as gbr

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

#Wine_DATA
filepath = os.sep.join(data_path + ['Ames_Housing_Sales.csv'])
data = pd.read_csv(filepath, sep=',')
result = pd.Series()
X_data = data.copy()
y_data = X_data.pop('SalePrice')

string_columns = X_data.dtypes
string_boolidx = string_columns == np.object

tr_data_num = X_data.drop(X_data.columns[string_boolidx], axis=1)

X_train, X_test, y_train, y_test = train_test_split(tr_data_num, y_data, test_size=0.3)

params = {'n_estimators':3000, 'max_depth':5, 'learning_rate':0.009, 'loss':'ls', 'min_samples_split':20, 'min_samples_leaf':3, 'max_features':'sqrt', 'random_state':0}

clf2 = gbr(**params)
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
print('HousingSale best case : ',rmse(y_test, y_pred))

#Testing 10 times
print("---Testing 10 times---")

for k in range(10):
   X_train, X_test, y_train, y_test = train_test_split(tr_data_num, y_data, test_size=0.3)
   params = {'n_estimators':3000, 'max_depth':5, 'learning_rate':0.009, 'loss':'ls', 'min_samples_split':20, 'min_samples_leaf':3, 'max_features':'sqrt', 'random_state':0}
    
   clf2 = gbr(**params)
   clf2.fit(X_train, y_train)
   y_pred = clf2.predict(X_test)
   result.at[k] = rmse(y_test,y_pred)
   print('rmse Result  :  ',rmse(y_test,y_pred))

print('10times rmse mean : ',result.mean())
