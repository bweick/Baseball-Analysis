# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:22:57 2015

@author: Brian
"""

import scrape_tools
import bayes_calc
import json
import pandas as pd
import numpy as np
from time import strftime

day = strftime("%Y-%m-%d")
desc = 'testAfternoon'
pitchpath = 'C:\\Users\Brian\\Documents\\Predictions\\MLB\\Pitcher' + desc +'_' + day + '.csv' 
batpath = 'C:\\Users\Brian\\Documents\\Predictions\\MLB\\Batter' + desc +'_' + day + '.csv'

json_file = open('C:\\Users\\Brian\\Documents\\Player Dictionaries\\MLB\\batters.json')
json_str = json_file.read()
batters = json.loads(json_str)
json_file.close()

json_file = open('C:\\Users\\Brian\\Documents\\Player Dictionaries\\MLB\\pitchers.json')
json_str = json_file.read()
pitchers = json.loads(json_str)
json_file.close()

json_file = open('C:\\Users\\Brian\\Documents\\Player Dictionaries\\MLB\\lineups.json')
json_str = json_file.read()
lineups = json.loads(json_str)
json_file.close()

parkf = pd.read_csv('C:\Users\Brian\Documents\Player Dictionaries\MLB\Park Factors Split.csv')
del parkf['Season']

sparkf = pd.read_csv('C:\Users\Brian\Documents\Player Dictionaries\MLB\Park Factors.csv')
del sparkf['Season']

print('Unpacking batter dictionaries...')
ratiob = scrape_tools.jsonopen('...\\MLB\\ratio_batter.json',
                               ['Year', 'Tm', 'PA', 'HR%', 'SO%', 'BB%', 'XBH%', 'X/H%', 'SO/W', 'AB/SO', 'AB/HR', 'GB/FB', 'IP%', 'LD%', 'HR/FB', 'IF/FB', 'GB%', 'FB%'])
batp15 = scrape_tools.jsonopen('...\\MLB\\bat_platoon_2015.json',
                               ['Split', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'SB', 'SO', 'BB', 'BAbip'])
batp14 = scrape_tools.jsonopen('...\\MLB\\bat_platoon_2014.json',
                               ['Split', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'SB', 'SO', 'BB', 'BAbip'])
print('Half-way through...')
batp13 = scrape_tools.jsonopen('...\\MLB\\bat_platoon_2013.json',
                               ['Split', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'SB', 'SO', 'BB', 'BAbip'])
batp12 = scrape_tools.jsonopen('...\\MLB\\bat_platoon_2012.json',
                               ['Split', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'SB', 'SO', 'BB', 'BAbip'])
normbatp = {}
for key in batters:
    print key
    if key in pitchers: ##forgo pitchers for now due to crappy data
        None
    else:            
        if batters[key][3] == 'R':
            arm = 0
        else:
            arm = 1
        comb = bayes_calc.YearCheck(key, arm, batp12, batp13, batp14, batp15)
        time = len(comb)
        yrs = [2015 - i for i in range(0,time)]
        comb['xBAbip'], comb['Team'] = bayes_calc.xbabip(ratiob, key, yrs)
        if batters[key][4] == 'H':
            stad = batters[key][0]
        else:
            stad = batters[key][1]
        if (batters[key][2] == 'R') or (batters[key][2] == 'B' and batters[key][3] == 'L'):
            tempf = parkf.reindex(columns=['Team', '1BR', '2BR', '3BR', 'HRR'])
            tempf = tempf.rename(columns = {'1BR':'Single', '2BR': 'Double', '3BR':'Triple', 'HRR':'Homer'})
            stempf = sparkf[sparkf['Team'] == stad].reindex(columns=['Team', '1BR', '2BR', '3BR', 'HRR'])
            stempf = stempf.rename(columns = {'1BR':'SingleC', '2BR': 'DoubleC', '3BR':'TripleC', 'HRR':'HomerC'})
        if (batters[key][2] == 'L') or (batters[key][2] == 'B' and batters[key][3] == 'R'):
            tempf = parkf.reindex(columns=['Team', '1BL', '2BL', '3BL', 'HRL'])
            tempf = tempf.rename(columns = {'1BL':'Single', '2BL': 'Double', '3BL':'Triple', 'HRL':'Homer'})
            stempf = sparkf[sparkf['Team'] == stad].reindex(columns=['Team', '1BL', '2BL', '3BL', 'HRL'])
            stempf = stempf.rename(columns = {'1BL':'SingleC', '2BL': 'DoubleC', '3BL':'TripleC', 'HRL':'HomerC'})
        comb = pd.merge(comb, tempf, on = 'Team')
        comb = bayes_calc.PlatDataNorm(comb, stempf)
        comb['BIP'] = comb['PA'] - comb['HR'] - comb['BB'] - comb['SO']
        proj = bayes_calc.BatWSumProj(comb)
        hr, so, bb, bip = bayes_calc.BayesCalc(proj, batters[key][3], batters[key][2])
