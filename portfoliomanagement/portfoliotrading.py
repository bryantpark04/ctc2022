#!/usr/bin/env python
# coding: utf-8

# In[16]:


##get_ipython().system('pip install yfinance ')


# In[17]:


import math
import numpy as np
import pandas as pd 
import yfinance
import matplotlib.pyplot as plt
import scipy
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests


# In[34]:


def pairs_trading_checking(prices):
    tickers = ['YPA','AOX','KWE','VSR','VWR','ZNX','MWN','MZJ','VEO','GJA','TUW','HBM','WNF','ABM','RLR','GPE','BUL','PKU','VLK','VYQ','EVW','NQW','PHV','DAI','WGQ','JFL','RLF','NFR','ZKY','STK','OWG','YLF','CPM','TGI','MGW','NTR','KKD','JPN','CGS','VOX']
    ##prices = pd.read_csv("prices.csv")
    prices.head()
    stock_list = prices[1:]

    for x in stock_list:
        if x == 'Ticker':
            stock_list.pop(x)

    stock_prices = []

    for x in stock_list: 
        datatester = stock_list[x]
        stock_prices.append(datatester)

    pvalues = [] 

    for stock in stock_prices:
        stat = adfuller(stock)
        pvalues.append([stat[1],stock.name])

    nonstationary = []
    stationary = []

    for x in pvalues:
        if x[0] > .05:
            nonstationary.append(x[1])
        elif x[0] <= .05:
            stationary.append(x[1])
            
            
    diffvalues = []
    for name in nonstationary:
        for stock in stock_prices:
            if name == stock.name:
                diff = stock.diff()
                diff = diff[1:]
                diffvalues.append([diff, name])
            
    new_pvalues = [] 

    for stock in diffvalues:
        stat = adfuller(stock[0])
        new_pvalues.append([stat[1],stock[-1]])


    for x in new_pvalues:
        if x[0] <= .05:
            stationary.append(x[1])
    


# In[35]:


maxlag = 2
test = 'ssr_chi2test'
df = pd.read_csv("new_ordered_difference_data.csv")
df.head()

def grangerscausationmatrix(data, variables, test = 'ssr_chi2test'
                            , verbose = False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), variables, variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag, False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df



causation_matrix = grangerscausationmatrix(df, df.columns)


# In[38]:


tickers = ['YPA','AOX','KWE','VSR','VWR','ZNX','MWN','MZJ','VEO','GJA','TUW','HBM','WNF','ABM','RLR','GPE','BUL','PKU','VLK','VYQ','EVW','NQW','PHV','DAI','WGQ','JFL','RLF','NFR','ZKY','STK','OWG','YLF','CPM','TGI','MGW','NTR','KKD','JPN','CGS','VOX']

def strat_function1(causation_matrix, tickers):
    pairs_trading_checking()
    grangerscausationmatrix(df, df.columns)
    tickers_x = []
    for x in tickers:
        x = x+"_x"
        tickers_x.append(x)

    tickers_y = []
    for y in tickers:
        y = y+"_y"
        tickers_y.append(y)

    causation_list = []

    for x in tickers_x:
        current_max = 0
        acc = -1
        for num in causation_matrix[x]:
            if float(num) != 1 and float(num) > current_max:
                current_max = num 
        for num in causation_matrix[x]:
            acc = acc + 1
            if num == current_max:
                real_acc = acc
        causation_list.append([x, current_max, tickers_y[real_acc]])
    
    stockpairs = []
    for list in causation_list:
        if list[1] >= 0.95:
            stockpairs.append([list[0], list[2]])

    return(stockpairs)
emma=[]

def strat_function(preds, prices, last_weights):
    weights=[]
    if prices.length()==2:
        for i in range(20):
            emma.append(prices[i+1][2])
    if prices.length()<50:
        for i in range(20):
            weights.append(.05)
    elif prices.length()==50:
        df=pd.DataFrame(prices)
        pairs_trading_checking(df)
        weights=last_weights
        causation_matrix = grangerscausationmatrix(df, df.columns)
        pairs=strat_function1(causation_matrix,tickers)
    else:
        weights=[]
    return weights












