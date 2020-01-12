#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 01:34:01 2019

@author: nick
"""

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import precision_score, recall_score, accuracy_score
from merging_for_model import merge_data

points = 13
my_data = pd.read_csv('/home/nick/Desktop/DF_simulator/build_model/output.csv', delimiter=',')
X_train = my_data.drop(['Points','Player'], axis = 1)
my_data.loc[my_data['Points'] < points, 'Points'] = 0
my_data.loc[my_data['Points'] >= points, 'Points'] = 1
Y_train = my_data['Points']


poly = PolynomialFeatures(interaction_only=True,include_bias = True)
X_train = poly.fit_transform(X_train)


D_train = xgb.DMatrix(X_train, label=Y_train)
#D_test = xgb.DMatrix(X_test, label=Y_test)

prediction_data = merge_data(21)
prediction_data.to_csv("build_model/merged15.csv", index=False)
players = prediction_data['Player']
X_test = prediction_data.drop(['Points','Player'], axis = 1)

prediction_data.loc[prediction_data['Points'] < points, 'Points'] = 0
prediction_data.loc[prediction_data['Points'] >= points, 'Points'] = 1
Y_test = prediction_data['Points']

poly = PolynomialFeatures(interaction_only=True,include_bias = True)
X_test = poly.fit_transform(X_test)

D_test = xgb.DMatrix(X_test, label=Y_test)


param = {
    'eta': 0.01, 
    'max_depth': 5,  
    'objective': 'multi:softprob',  
    'num_class': 2} 

steps = 1000 

param_2 = {
    'eta': 0.001, 
    'max_depth': 5,
    
    'booster': 'gbliner',
    'objective': 'multi:softprob',  
    'num_class': 2}

model = xgb.train(param, D_train, steps)
#model = xgb.XGBRegressor(param_2, D_train, steps)

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))

players = pd.DataFrame(players)
result = players.join(pd.DataFrame(preds))

result.to_csv("build_model/result.csv", index=False)






