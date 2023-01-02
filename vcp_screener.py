
"""
This module could be run in Quantconnect Lean Engine for backtesting.

It is a stock screener for US equities based on VCP strategy.

@author: mickeylau
"""
from AlgorithmImports import *
from System.Collections.Generic import List
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class VCP2(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        self.SetStartDate(2018,1,1)  #Set Start Date
        self.SetEndDate(2022,12,31)    #Set End Date
        self.SetCash(100000)           #Set Strategy Cash
        self.SetWarmUp(400, Resolution.Daily)


        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.Leverage = 1.5 #allow some leverage due to limitation of resolution

        self.coarse_count = 10 #initial number of stocks

        self.SetExecution(ImmediateExecutionModel())
        self.SetRiskManagement(NullRiskManagementModel())
        self.averages = { }

        # add universe method
        # coarse selection function: accepts an IEnumerable<CoarseFundamental> and returns an IEnumerable<Symbol>
        self.AddUniverse(self.CoarseSelectionFunction)

  
    # screen data and take the top 'NumberOfSymbols'
    def CoarseSelectionFunction(self, coarse):

        self.filtered_by_price = [x for x in coarse if x.Price>15 and x.DollarVolume>500000 and x.HasFundamentalData]

        # use a dictionary to refer the object that will keep the moving averages
        for cf in self.filtered_by_price:

           
            if cf.Symbol not in self.averages:
                self.averages[cf.Symbol] = SymbolData(self, cf.Symbol)

            # Updates the SymbolData object with current EOD price
            avg = self.averages[cf.Symbol]
            avg.update(cf.EndTime, cf.AdjustedPrice)

        # Filter the values of the dict: we only want up-trending securities
        values = list(filter(lambda x: x.is_uptrend, self.averages.values()))

        # Sorts the values of the dict: we want those with greater slope
        values.sort(key=lambda x: x.scale, reverse=True)

        for x in values[:self.coarse_count]:
            self.Log('symbol: ' + str(x.symbol.Value) + '  scale: ' + str(x.scale))

        # we need to return only the symbol objects
        return [ x.symbol for x in values[:self.coarse_count]]

    # this event fires whenever we have changes to our universe
    def OnSecuritiesChanged(self, changes):
        # liquidate removed securities
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)

        # we want 10% invested in each security selected
        for security in changes.AddedSecurities:
                self.SetHoldings(security.Symbol, 0.1) 



class SymbolData(object):
    def __init__(self, algorithm, symbol):
        self.symbol = symbol
        self.MA10 = ExponentialMovingAverage(10)
        self.MA20 = ExponentialMovingAverage(20)
        self.MA50 = ExponentialMovingAverage(50)
        self.MA200 = ExponentialMovingAverage(200)
        self.lookback = int(252/2)

        self.is_uptrend = False
        self.scale = 0
        self.slope=0


        history = algorithm.History(self.symbol, self.lookback, Resolution.Daily)
        current = algorithm.History(self.symbol, 28, Resolution.Minute)

        self.price = {}
     
    
        if not history.empty and not current.empty:
            self.price[symbol.Value] = list(history.loc[self.symbol]['open'])
            self.price[symbol.Value].append(current.loc[self.symbol]['open'][0])

        A = range( self.lookback + 1 )
        if symbol.Value in self.price:
            # volatility
            std = np.std(self.price[symbol.Value])
            # Price points to run regression
            Y = self.price[symbol.Value]
            # Add column of ones so we get intercept
            X = np.column_stack([np.ones(len(A)), A])
            if len(X) != len(Y):
                length = min(len(X), len(Y))
                X = X[-length:]
                Y = Y[-length:]
                A = A[-length:]
            # Creating Model
            reg = LinearRegression()
            # Fitting training data
            
            reg = reg.fit(X, Y)
            # run linear regression y = ax + b
            b = reg.intercept_
            a = reg.coef_[1]
            
            # Normalized slope
            slope = a / b *252.0

            self.slope = slope


    def update(self, time, value):
        if self.MA10.Update(time, value) and self.MA20.Update(time, value) and self.MA50.Update(time, value) and self.MA200.Update(time, value):
            MA10 = self.MA10.Current.Value
            MA20 = self.MA20.Current.Value
            MA50 = self.MA50.Current.Value
            MA200 = self.MA200.Current.Value
            self.is_uptrend = MA10 > MA50 and MA20>MA200 and MA50>MA200

        if self.is_uptrend:
            self.scale = self.slope
