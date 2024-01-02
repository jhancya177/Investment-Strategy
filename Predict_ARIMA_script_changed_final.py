# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:35:48 2022

@author: ASUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
import itertools


start = datetime.date.today() - relativedelta(years=2)
end = datetime.date.today() 
symbolscsv = pd.read_csv("C:\\Users\\shikh\\Downloads\\symbols_final.csv")
symbols = symbolscsv['symbols'].values.tolist()

stock_list = []
price_list = []
forecast_list=[]
orig_close_list=[]
for stock in symbols:
    stock_close = yf.download(stock,start=start,end = end, interval='1d')['Close']
    stock_close.dropna(inplace = True)
    arima_T_pred = ARIMA(stock_close,order=(3,1,2))

    arima_fitted_T_pred= arima_T_pred.fit()

    fc = arima_fitted_T_pred.forecast(5, alpha=0.05)
    
    stock_list.append(stock)
    print(fc)
    fc1 = pd.DataFrame(fc).T.values[0]
   
    forecast_list.append(fc1)
    print(forecast_list)
result = pd.concat([pd.Series(stock_list) , pd.Series(forecast_list)], axis=1)
result.to_csv('Arima_prediction.csv')
