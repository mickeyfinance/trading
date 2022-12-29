#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:19:45 2022

@author: mickeylau
"""

import yfinance as yf
from finviz.screener import Screener
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import linregress
from IPython.display import Image, display
from IPython.core.display import HTML
from finvizfinance.quote import finvizfinance
import statsmodels.api as sm

def slope(ser,n):
    "function to calculate the slope of n consecutive points on a plot"
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y = ser[i-n:i]
        x = np.array(range(n))
        y_scaled = (y - y.min())/(y.max() - y.min())
        x_scaled = (x - x.min())/(x.max() - x.min())
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

    
def slope_reg(arr):
    y = np.array(arr)
    x = np.array(arr)
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    print(arr)
    return slope
    
    
def stock_filter(df,a=30, b=50, c=150, d=200):

    df["MA_30"] = df["Close"].ewm(span=a,min_periods=a).mean()
    df["MA_50"] = df["Close"].ewm(span=b,min_periods=b).mean()
    df["MA_150"] = df["Close"].ewm(span=c,min_periods=c).mean()
    df["MA_200"] = df["Close"].ewm(span=d,min_periods=d).mean()
    
    df['MA_slope_200'] = slope(df['MA_200'],20)
    df['MA_slope_30'] = slope(df['MA_30'],20)
    
    df['52_week_low'] = df['Close'].rolling(window= 5*52).min()
    df['52_week_high'] = df['Close'].rolling(window= 5*52).max()
    
    
    #Criteria 1: the current price of the security must be greater than the 150 and 200 day simple moving average    
    df["Criteria1"] = (df["Close"] > df["MA_150"]) & (df["Close"] > df["MA_200"])

    # Criteria 2: 150 day must be greater than 200
    df['Criteria2'] = (df['MA_150'] > df['MA_200'])
    
    #Criteria 3 : 200 day must be trending up for at least 1 month
    df["Criteria3"] = df['MA_slope_200'] >0.0
    
    #Criteria 4 : 50 MA > 150 MA and 150 > 200
    df["Criteria4"] = (df['MA_50'] > df['MA_150']) & (df['MA_150']>df['MA_200'])
    
    #Criteria 5: The current price > 50-day
    df['Criteria5'] = (df['Close'] > df['MA_50'])
    
    #Criteria 6: Current price at least 30% above 52 week low
    df['Criteria6'] = (df['Close'] - df['52_week_low'])/df['52_week_low']>0.3
    
    #Criteria 7: The current price must be within 15% of the 52 week high
    df['Criteria7'] = ((df['Close'] - df['52_week_high'])/df['52_week_high'] < 0.15) & ((df['Close'] - df['52_week_high'])/df['52_week_high']>-0.15)
        
    # Criteria 8 : proxy of RS ratio
    df['Criteria8'] = (df['Close'] - df['Close'].shift(periods=250))/df['Close'].shift(periods=250) > 0.89
    
    #Criter 9: Need to breakout Pivot (5 day)
    df['Criteria9'] = df['Close'] > ( (df['Close']).rolling(window=5).mean() + (df['Close']).rolling(window=5).max() + (df['Close']).rolling(window=5).min())/3
    
    #Criteria 10 : Contraction below 10%
    df['Criteria10'] = ( (df['Close']).rolling(window=10).max() - (df['Close']).rolling(window=10).min()) / (df['Close']).rolling(window=10).min() <0.1
    
    # Criteria 11: 30 day must be trending up
    df["Criteria11"] = df['MA_slope_30'] >0.0
    
    # Check criteria
    df['Fullfilment'] = df[['Criteria1','Criteria2','Criteria3','Criteria4','Criteria5','Criteria6','Criteria7','Criteria8','Criteria9','Criteria10','Criteria11']].all(axis='columns')
    
    return df[['Close','Fullfilment']]
    

def test_vcp(df):
    df['Future_Close_1'] = df['Close'].shift(periods=-1)
    df['Future_Close_3'] = df['Close'].shift(periods=-3)
    df['Future_Close_5'] = df['Close'].shift(periods=-5)
    df['Future_Close_7'] = df['Close'].shift(periods=-7)
    df['Result_1'] = (df['Future_Close_1'] - df['Close']) / df['Close']
    df['Result_3'] = (df['Future_Close_3'] - df['Close']) / df['Close']
    df['Result_5'] = (df['Future_Close_5'] - df['Close']) / df['Close']
    df['Result_7'] = (df['Future_Close_7'] - df['Close']) / df['Close']
    vcp_date = df[df['Fullfilment'] == True]
    print(vcp_date[['Result_1', 'Result_3', 'Result_5', 'Result_7' ]])
    print(vcp_date[['Result_1', 'Result_3', 'Result_5', 'Result_7' ]].describe())

"""
def show_image(ticker):
    stock = finvizfinance(ticker)
    print(stock.TickerCharts())
    display(Image(url=stock.TickerCharts()))
"""


filters = ["cap_midover","sh_price_o20","fa_salesqoq_o5","sh_instown_o10"]
stock_list=Screener(filters = filters, table = "Performance", order= "Price")

ticker_table= pd.DataFrame(stock_list.data)
ticker_list=ticker_table["Ticker"].to_list()    
    
    
for ticker_string in tqdm(ticker_list):
    ticker = yf.Ticker (ticker_string)
    ticker_history = ticker.history(period='max')
    data = stock_filter(ticker_history)
    if data['Fullfilment'].tail(1).iloc[0] == True:
        print (ticker_string)
        print (data)
        test_vcp(data)
    #   show_image(ticker_string)

        
        
    
    
    