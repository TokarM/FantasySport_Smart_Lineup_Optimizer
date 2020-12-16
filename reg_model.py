#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:48:21 2019

@author: nick
"""
import numpy
import pandas
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from merging_for_model import merge_data

'''
# Predict poinst using linear regresion
def reg_points(week, test_week):
    players_csv = merging.merging_ucl(week)
    players_csv = players_csv.replace('-', numpy.NaN)
    for index, row in players_csv.iterrows():
    
        if pandas.isna(row[test_week]):
            players_csv.at[index, test_week] = row['AVG FPTS']
        
    players_csv['Percentage'] = players_csv['Percentage'] / 100
    
    X = players_csv[['Salary','Percentage','G','A', 'SOG', 'INT', 'CR', 'TKLW', 'P', 'FS', 'FC', 'S', 'CC']]
    poly = PolynomialFeatures(interaction_only=True,include_bias = True, degree = 3)
    X = poly.fit_transform(X)
    
    y = players_csv[test_week]
    
    reg = LinearRegression(normalize = True).fit(X,y)
    score = reg.score(X,y)
    prediction = reg.predict(X)
    
    prediction = pandas.DataFrame(prediction)
    prediction['Player'] = players_csv['Player']
    prediction = prediction.set_index('Player')
    
    return prediction

week = '12.9'
test_week = '5'
players_csv = merging.merging_ucl(week)
players_csv = players_csv.replace('-', numpy.NaN)
for index, row in players_csv.iterrows():

    if pandas.isna(row[test_week]):
        players_csv.at[index, test_week] = row['AVG FPTS']
    
players_csv['Percentage'] = players_csv['Percentage'] / 100
'''

my_data = pandas.read_csv('build_model/output.csv', delimiter=',')
my_data = my_data[['Player','Salary', 'S', 'INT', 'CR', 'CC', 'FC','P','Total_Kicks', 'Win %', 'D', 'M','F',
                   'LIV', 'LEI', 'CHE', 'TOT', 'BHA', 'AVL','BOU', 'SOU', 'Points']]
X_train = my_data.drop(['Points','Player'], axis = 1)
#my_data.loc[my_data['Points'] < 13, 'Points'] = 0
#my_data.loc[my_data['Points'] >= 13, 'Points'] = 1
Y_train = my_data['Points']

poly = PolynomialFeatures(interaction_only=True,include_bias=True, degree=3)
X = poly.fit_transform(X_train)

y = Y_train 

# Specify week of prediction
my_data15 = merge_data(19)

my_data15 = my_data15[['Player','Salary','S', 'INT', 'CR', 'CC', 'FC','P','Total_Kicks', 'Win %', 'D', 'M','F',
                   'LIV', 'LEI', 'CHE', 'TOT', 'BHA', 'AVL','BOU', 'SOU', 'Points']]
players = my_data15['Player']
X_test = my_data15.drop(['Points','Player'], axis = 1)

my_data15.loc[my_data15['Points'] < 13, 'Points'] = 0
my_data15.loc[my_data15['Points'] >= 13, 'Points'] = 1
Y_test = my_data15['Points']

poly = PolynomialFeatures(interaction_only=True,include_bias=True, degree=3)
X_test = poly.fit_transform(X_test)


reg = LinearRegression(normalize = True).fit(X,y)
score = reg.score(X,y)
prediction = reg.predict(X_test)
prediction = pandas.DataFrame(prediction)

prediction['Player'] = players
#prediction = prediction.set_index('Player')







