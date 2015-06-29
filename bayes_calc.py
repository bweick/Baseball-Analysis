# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 18:21:55 2015

@author: Brian
"""
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt

def YearCheck(key, arm, dic1, dic2, dic3, dic4):
    if key in dic1:
        dic1[key]['Split'] = 2012
        temp12 = dic1[key].loc[arm, :]
    else:
        temp12 = pd.DataFrame(columns = ['Split', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'SB', 'SO', 'BB', 'BAbip'])
    if key in dic2:
        dic2[key]['Split'] = 2013
        temp13 = dic2[key].loc[arm, :]
    else:
        temp13 = pd.DataFrame(columns = ['Split', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'SB', 'SO', 'BB', 'BAbip'])
    if key in dic3:
        dic3[key]['Split'] = 2014
        temp14 = dic3[key].loc[arm, :]
    else:
        temp14 = pd.DataFrame(columns = ['Split', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'SB', 'SO', 'BB', 'BAbip'])
    if key in dic4:
        dic4[key]['Split'] = 2015
        temp15 = dic4[key].loc[arm, :]
    else:
        temp15 = pd.DataFrame(columns = ['Split', 'PA', 'AB', 'H', '2B', '3B', 'HR', 'SB', 'SO', 'BB', 'BAbip'])
    comb = pd.concat([temp12, temp13, temp14, temp15], axis = 1).T
    comb = comb.drop_duplicates(subset=['PA', 'H', 'BAbip'], take_last = True)
    comb = comb.dropna(how = 'all')
    comb = comb.reset_index(drop=True)
    return comb

def xbabip(dic, name, yrs):
    df = dic[name]
    df = df[df['Year'].isin(yrs)]
    df['xbip'] =  df['GB%']*.24 + df['FB%']*(1-df['IF/FB'])*.15 + df['LD%']*.73  
    df = df.reset_index()
    return df['xbip'], df['Tm']

def PlatDataNorm(comb, stempf):
    hrc = int(stempf['HomerC'])
    comb['HR'] = (comb['HR'])/(comb['Homer']/hrc)
    comb['HR'] = comb['HR'].apply(np.round)
#    comb['1B'] = (comb['H'] - comb['2B'] - comb['3B'] - comb['HR'])
#    comb['SB'] = comb['SB']/(comb['1B'] + comb['BB'])
#    comb['1B'] = (comb['H'] - comb['2B'] - comb['3B'] - comb['HR'])/(comb['H']-comb['HR'])
#    comb['2B'] = comb['2B']/(comb['H']-comb['HR'])
#    comb['3B'] = comb['3B']/(comb['H']-comb['HR'])
#    comb['BB'] = comb['BB']/comb['PA']
#    comb['SO'] = comb['SO']/comb['PA']
#    comb['x1B'] = (comb['1B']/comb['Single'])/((comb['1B']/comb['Single'])+(comb['2B']/comb['Double'])+(comb['3B']/comb['Triple']))
#    comb['x2B'] = (comb['2B']/comb['Double'])/((comb['1B']/comb['Single'])+(comb['2B']/comb['Double'])+(comb['3B']/comb['Triple']))
#    comb['x3B'] = (comb['3B']/comb['Triple'])/((comb['1B']/comb['Single'])+(comb['2B']/comb['Double'])+(comb['3B']/comb['Triple']))
#    Hsum = np.sum(comb['H'])
#    PAsum = np.sum(comb['PA'])
#    ABsum = np.sum(comb['AB'])
#    comb['wPA'] = comb['PA']/PAsum 
#    comb['wAB'] = comb['AB']/ABsum
#    comb['wH'] = comb['H']/Hsum
    return comb
    
def BatWSumProj(df):
    df = np.sum(df, axis = 0)
    return df

def HRBayes(pa, hr, alpha, beta):
    hrpg = np.zeros(pa)
    hrpg[:hr] = 1
    phr = pm.Beta("phr", alpha, beta)
    obshr = pm.Bernoulli("obshr", phr, value = hrpg, observed = True)
    mcmc = pm.MCMC([phr, obshr])
    mcmc.sample(20000, 7000)
    sumstats = mcmc.stats()
    loint = sumstats['phr']['95% HPD interval'][0]
    hiint = sumstats['phr']['95% HPD interval'][1]
    hr_trace = mcmc.trace('phr')[:]
    hr_trace = [x for x in hr_trace if (x > loint) and (x < hiint)]
    return hr_trace
    
def SOBayes(pa, so, mu, tau):
    sopg = np.zeros(pa)
    sopg[:so] = 1
    pso = pm.Normal("pso", mu, tau)
    obsso = pm.Bernoulli("obsso", pso, value = sopg, observed = True)
    mcmc = pm.MCMC([pso, obsso])
    mcmc.sample(20000, 7000)
    sumstats = mcmc.stats()
    loint = sumstats['pso']['95% HPD interval'][0]
    hiint = sumstats['pso']['95% HPD interval'][1]
    so_trace = mcmc.trace('pso')[:]
    so_trace = [x for x in so_trace if (x > loint) and (x < hiint)]
    return so_trace

def BBBayes(pa, bb, mu, tau):
    bbpg = np.zeros(pa)
    bbpg[:bb] = 1
    pbb = pm.Normal("pbb", mu, tau)
    obsbb = pm.Bernoulli("obsbb", pbb, value = bbpg, observed = True)
    mcmc = pm.MCMC([pbb, obsbb])
    mcmc.sample(20000, 7000)
    sumstats = mcmc.stats()
    loint = sumstats['pbb']['95% HPD interval'][0]
    hiint = sumstats['pbb']['95% HPD interval'][1]
    bb_trace = mcmc.trace('pbb')[:]
    bb_trace = [x for x in bb_trace if (x > loint) and (x < hiint)]
    return bb_trace

def BIPBayes(pa, bip, mu, tau):
    bippg = np.zeros(pa)
    bippg[:bip] = 1
    pbip = pm.Normal("pbip", mu, tau)
    obsbip = pm.Bernoulli("obsbip", pbip, value = bippg, observed = True)
    mcmc = pm.MCMC([pbip, obsbip])
    mcmc.sample(20000, 7000)
    sumstats = mcmc.stats()
    loint = sumstats['pbip']['95% HPD interval'][0]
    hiint = sumstats['pbip']['95% HPD interval'][1]
    bip_trace = mcmc.trace('pbip')[:]
    bip_trace = [x for x in bip_trace if (x > loint) and (x < hiint)]
    return bip_trace
    

def BayesCalc(df, arm, bat): ###very messy code, could be much nicer 
    if arm == 'R' and bat == 'R':
        hr_trace = HRBayes(df['PA'], df['HR'], .416, 17.3)
        so_trace = SOBayes(df['PA'], df['SO'], .209, 227)
        bb_trace = BBBayes(df['PA'], df['BB'], .069, 1130)
        bip_trace = BIPBayes(df['PA'], df['BIP'], .7, 146)
    if arm =='L' and bat in ['R', 'B']:
        hr_trace = HRBayes(df['PA'], df['HR'], .45, 15.3)
        so_trace = SOBayes(df['PA'], df['SO'], .184, 274)
        bb_trace = BBBayes(df['PA'], df['BB'], .089, 497)
        bip_trace = BIPBayes(df['PA'], df['BIP'], .7, 139.5)
    if arm =='R' and bat in ['L', 'B']:
        hr_trace = HRBayes(df['PA'], df['HR'], .52, 22.7)
        so_trace = SOBayes(df['PA'], df['SO'], .215, 246)
        bb_trace = BBBayes(df['PA'], df['BB'], .086, 1057)
        bip_trace = BIPBayes(df['PA'], df['BIP'], .71, 216)
    if arm =='L' and bat == 'L':
        hr_trace = HRBayes(df['PA'], df['HR'], .176, 10.75)
        so_trace = SOBayes(df['PA'], df['SO'], .181, 418.5)
        bb_trace = BBBayes(df['PA'], df['BB'], .067, 980)
        bip_trace = BIPBayes(df['PA'], df['BIP'], .699, 147.5)
    return hr_trace, so_trace, bb_trace, bip_trace