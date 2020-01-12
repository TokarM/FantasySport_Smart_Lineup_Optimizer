#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:22:57 2019

@author: nick
"""

import numpy
import pandas
import df_simulator_func 
from operator import add
from operator import truediv
from scipy.stats import poisson
from scipy.stats import nbinom
from itertools import combinations
import merging
import reg_model


# Read Data
#players_csv = merging.merging_epl('16.1')
#players_csv = players_csv[players_csv['Injury Indicator'] != 'O']


def read_data(champ, week, minutes = 500, points=5):
    if champ == 'epl':
        players_csv = merging.merging_epl(week)
    elif champ == 'ucl':
        players_csv = merging.merging_ucl(week)
    players_csv = players_csv[players_csv['FPPG'] > points]
    players_csv = players_csv[players_csv['TOTAL MINS'] > minutes]
    return players_csv

def create_matrix(players_csv):
    players_matrix = numpy.zeros(10000)
    for index, row in players_csv.iterrows():
        players_matrix = numpy.vstack((players_matrix,(poisson.rvs(float(row['FPPG']), size = 10000))))
        
    players_matrix = numpy.delete(players_matrix, 0,0)
    return players_matrix

def create_dict(players_csv):
    players_dict = {}
    for x,y in zip(players_csv['Name'], range(len(players_csv))):
        players_dict.update({x:y})

def calc_points(player):
    total = 0
    total += player.goal_chance * 12 + ((player.goal_chance**2) * 12) + ((player.goal_chance**3) * 12)
    total += player.assist * 6
    total += player.s * 1
    total += player.sog * 2
    total += player.interception * 0.5
    total += player.fs * 1
    total += player.fc * 1.3
    total += player.tklw * 1
    total += player.p * 0.02
    total += player.cr * 0.7
    total += player.cc * 1
    return total
    
   
# Create lineups using Maximum expected utility function
def meu_lineups(gk,def_,fwd, min_limit = 40000, max_limit = 45000):
    lineups = []
    for i in combinations(range(len(fwd)), 2):
        print(i)
        lineup = []
        
        for j in combinations(range(len(def_)),2): 
            for k in combinations(range(len(med)),2):
                cap = 0
                meu = 0
                lineup = [] 
                # Add forwards
                for l in i:
                    cap += fwd[l].salary
                    points = 0
                    points = calc_points(fwd[l])
                    meu += ((points * fwd[l].prob) + ((-fwd[l].salary/50) * (1 - fwd[l].prob)))
                    lineup.append(fwd[l].name)
                # Add defenders
                for m in j:
                    cap += def_[m].salary
                    points = 0
                    points = calc_points(def_[m])
                    meu += (points * def_[m].prob + (-def_[m].salary/50) * (1 - def_[m].prob))
                    lineup.append(def_[m].name)
                # Add mid players
                for n in k:
                    cap += med[n].salary
                    points = 0
                    points = calc_points(med[n])
                    meu += (points * med[n].prob + (-med[n].salary/50) * (1 - med[n].prob))
                    lineup.append(med[n].name)
                # Add util
                '''
                cap += players[u].salary
                points = 0
                points = calc_points(players[u])
                meu += (points * players[u].prob + (-players[u].salary) * (1 - players[u].prob))
                lineup.append(player[u].name)
                '''
                
                if cap >= min_limit and cap <= max_limit and len(set(lineup)) == 6:
                    lineups.append([lineup, cap, meu])
                
    lineups.sort(key = lambda x: x[2], reverse = True)
    team = pandas.DataFrame(lineups)
    return team
  
# Create lineups using Maximum expected utility function
def meu_lineups_reg(gk,def_,fwd, price_diff = 0, forwards=2, middle=2, defenders=2, num_players=7):
    
    min_limit = 40000
    max_limit = 45500
    
    min_limit = min_limit - price_diff
    max_limit = max_limit - price_diff
    
    lineups = []
    util = players[0:10]
    for i in combinations(range(len(fwd)), forwards):
        print(i)
        lineup = []
        
        for j in combinations(range(len(def_)),defenders): 
            for k in combinations(range(len(med)),middle):
                for q in range(len(util)):
                    cap = 0
                    meu = 0
                    lineup = [] 
                    # Add forwards
                    
                    for l in i:
                        cap += fwd[l].salary
                        meu += ((fwd[l].points * fwd[l].prob) + ((-fwd[l].salary/500) * (1 - fwd[l].prob)))
                        lineup.append(fwd[l].name)
                    
                    # Add defenders
                    for m in j:
                        cap += def_[m].salary
                        meu += (def_[m].points * def_[m].prob + (-def_[m].salary/500) * (1 - def_[m].prob))
                        lineup.append(def_[m].name)
                        
                    # Add mid players
                    for n in k:
                        cap += med[n].salary
                        meu += (med[n].points * med[n].prob + (-med[n].salary/500) * (1 - med[n].prob))
                        lineup.append(med[n].name)
                        
                    # Add util
                    cap += util[q].salary
                    meu += (util[q].points * util[q].prob + (-util[q].salary/500) * (1 - util[q].prob))
                    lineup.append(util[q].name)
                    
                    if cap >= min_limit and cap <= max_limit and len(set(lineup)) == num_players:
                        lineups.append([lineup, cap, meu])
                
    lineups.sort(key = lambda x: x[2], reverse = True)
    team = pandas.DataFrame(lineups)
    return team 

# Showdown meu   
def meu_showdown(players, min_limit = 40000, max_limit = 45000):
    lineups = []
    for i in combinations(range(len(players)), 6):
        print(i)
        cap = 0
        meu = 0
        lineup = []
        for u in i:
            cap += players[u].salary
            points = 0
            points = calc_points(players[u])
            meu += (points * players[u].prob + (-(players[u].salary / 50)) * (1 - players[u].prob))
            lineup.append(players[u].name)
            
            
            if cap <= max_limit and cap >= min_limit:
                    lineups.append([lineup, cap, meu])
        print(lineup)  
        print(cap)

    lineups.sort(key = lambda x: x[2], reverse = True)
    team = pandas.DataFrame(lineups)
    return team

# Read Players    
results_csv =  pandas.read_csv('/home/nick/Desktop/DF_simulator/build_model/result.csv', delimiter=',')
excepting = []
players_csv = read_data('epl', '21', minutes = 800, points = 8)

players = df_simulator_func.read_players(players_csv,results_csv, excepting, tour = 15)

#prediction = reg_model.reg_points('12.9', '5')
#players = df_simulator_func.read_players_prob(players_csv,excepting, tour = 5)
# Segregate DEF, GK, FWD

gk = []
def_ = []
med = []
fwd = []

for player in players:
    if player.position == 'F':
        fwd.append(player)
    elif player.position == 'M':
        med.append(player)
    elif player.position == 'D':
        def_.append(player)
    elif player.position == 'M/F':
        fwd.append(player)
        med.append(player)
    else:
        gk.append(player)


#team = meu_lineups(gk, def_, fwd, min_limit = 38000, max_limit = 42000)
#team = meu_showdown(players, max_limit = 50000)
team_avg = meu_lineups_reg(gk, def_, fwd, price_diff=9000, forwards=1, num_players=6)
team_1000 = team_avg[:1000]
        
#team_boost = meu_prebuild(gk, def_, fwd, min_limit = 40000, max_limit = 45500)
'''
results = pandas.read_csv('/home/nick/Desktop/DF_simulator/build_model/players_points.csv')

results = results[['PLAYER', '19']]
results['Player'] = results['PLAYER']  

l = team_avg[0].to_list()
team_results = []

team_results = df_simulator_func.test_lineups(l[0:100], results, '19')
team_results = pandas.DataFrame(team_results)
avg = numpy.mean(team_results[1])
'''



