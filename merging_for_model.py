#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:55:32 2019

@author: nick
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sympy
import scipy
import sklearn.neural_network as nn

import sklearn.metrics as metrics
import statsmodels.api as stats


team_dic = {'Liverpool':'LIV',
            'Leicester City':'LEI',
            'Manchester City':'MCI',
            'Chelsea':'CHE',
            'Manchester United':'MUN',
            'Wolverhampton':'WLV',
            'Tottenham Hotspur':'TOT',
            'Sheffield United':'SHU',
            'Arsenal': 'ARS',
            'Crystal Palace': 'CRY',
            'Newcastle United':'NEW',
            'Brighton & Hove Albion':'BHA',
            'Burnley':'BUR',
            'Everton':'EVE',
            'AFC Bournemouth':'BOU',
            'West Ham United':'WHU',
            'Aston Villa': 'AVL',
            'Southampton':'SOU',
            'Norwich City':'NOR',
            'Watford': 'WAT'}


def merge_data(week):
    directory = 'build_model/' + str(week)
    a = pd.read_csv(directory + "_players_stats.csv", header = 1)
    #a = a[['Name','Team','Salary','A', 'SOG', 'S', 'INT','CC','CR','FC', 'FS','TKLW', 'P']]
    a = a.drop(['MIN'], axis = 1)
    a['Player'] = a['Name']
    a = a.drop(['Name'], axis = 1)
    
    if isinstance(week, int):
        b = pd.read_csv(directory[:-2] + "/players_points.csv")
    else:
        b = pd.read_csv(directory[:-4] + "/players_points.csv")
    #b = b.iloc[:, :29]
    b['Player'] = b['PLAYER']
    b[str(int(week))] =b[str(int(week))].replace('-',0)
    
    c = pd.read_csv(directory + "_goal_odds.csv", header = 1)
    c = c[['Player','Percentage']]
    
    d = pd.read_csv(directory + "_free_kicks.csv", header = 1)
    d = d[['Total', 'Player']]
    
    e = pd.read_csv(directory + "_teams_odds.csv", header = 1)
    e['Team'] = e['Team'].map(team_dic)
    e = e[['Team','Win %']]
    
    # Merge players stats and points
    
    merged = a.merge(b[['Player', str(int(week))]], on='Player')
    merged = merged[merged[str(int(week))] != '-']
    
   
    # Create points column
    merged['Points'] = merged[str(int(week))]
    merged = merged.drop([str(int(week))], axis = 1)
    #merged = merged.drop(['Player'], axis = 1)
    merged = merged.dropna()
    # Merge goal odds
    merged = merged.merge(c, on='Player',how='left' )
    merged['Percentage'] = merged['Percentage'].fillna(0)
    
    # Merge free kicks
    merged = merged.merge(d,on='Player',how='left')
    merged['Total_Kicks'] = merged['Total'].fillna(0)
    merged['Total_Kicks'] = merged['Total_Kicks'].div(week)
    merged = merged.drop(['Total'], axis = 1)
    
    
    # Merge winning probability
    merged = merged.merge(e, on='Team', how = 'left')
    
    
    # Convert salary to int
    merged['Salary'] = merged['Salary'].str[1:]
    merged['Salary'] = merged['Salary'].astype(int)
    
    # Convert Points to Float
    merged['Points'] = merged['Points'].astype(float)
    
    # Create dummy var
    Position = pd.get_dummies(merged['Position'])
    Teams = pd.get_dummies(merged['Team'])
    
    # merge dummy var
    merged = merged.join(Position)
    
    #merged = merged.join(Teams)
    
    for i in list(team_dic.values()):
        merged[i] = 0
    
    for index, row in merged.iterrows():
        team = row['Team']
        merged.at[index, team] = 1
        
    # drop original H/A and Position
    merged = merged.drop(['Team','Position'], axis = 1)
    
    return  merged

merged = merge_data(6)


for i in range(7,20):
    merged = merged.append(merge_data(i), sort = False)
    
merged = merged.fillna(0)
merged['Index'] = range(0,len(merged))
merged = merged.set_index('Index')
merged.to_csv("build_model/output.csv", index=False)


#merge18 = merge_data(18)    


'''
X = merged.drop('Points', axis = 1)
poly = PolynomialFeatures(interaction_only=True,include_bias = False)
X = poly.fit_transform(X)

y = merged['Points'].astype('category')

reg = LinearRegression(normalize = True).fit(X,y)
score = reg.score(X,y)
prediction = reg.predict(X)
'''
