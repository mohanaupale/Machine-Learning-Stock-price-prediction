
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import sys
import math
import datetime as dt
import pandas_datareader as pdr
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import matplotlib.pylab as plt1
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf 

            
            

def load_edit_data():
    print ('\nEnter any company symbol from the following:\n [FB:Facebook, AAPL:Apple, GOOGL: Google](or 0 to quit):\n')
    choice = int(input('1:Google\n'
                           '2:Facebook \n'
                      '3:Quit'))
    if choice== 1:
        stockdata = pd.read_csv('WIKI-GOOGL.csv')
        print (stockdata.head())
        print ('\n The Features and their Data Types:\n',stockdata.dtypes)
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    
        stockdata = pd.read_csv('WIKI-GOOGL.csv', parse_dates=['Date'],index_col='Date',date_parser=dateparse)
       
    elif choice== 2 :
        stockdata = pd.read_csv('facebook.csv')
        print (stockdata.head())
        print ('\n The Features and their Data Types:\n',stockdata.dtypes)
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
   
        stockdata = pd.read_csv('facebook.csv', parse_dates=['Date'],index_col='Date',date_parser=dateparse)
      
    else:
        print('Please enter a valid input')
        
    return stockdata    
    l
                   
                   

    print (stockdata.head())
    #check datatype of index
    #print(data.index)
    return stockdata
    
    
def convert_timeseries(stockdata):
    #convert to time series:
    timeseries = stockdata['Adj Close']
    timeseries.head(10)
    print(timeseries['2017'])
    plt.plot(timeseries)
    return timeseries

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rollingmean = pd.rolling_mean(timeseries, window=20)
    rollingstd = pd.rolling_std(timeseries, window=20)

    #Plot rolling statistics:
    print("Test_stationary plot:")
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rollingmean, color='red', label='Rolling Mean')
    std = plt.plot(rollingstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['open','high','10_day_volatility', '50_day_moving_avg'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = math.ceil(value)
    print (dfoutput)

def convert_stationary(timeseries):
    timeseries_log = np.log(timeseries)
    plt.plot(timeseries_log)
    return timeseries_log

def data_smoothing(timeseries_log):    
    moving_avg = pd.rolling_mean(timeseries_log, 10, min_periods=1)
    print("In smoothing:")
    plt.figure(1) 
    plt.subplot(211)
    plt.plot(timeseries_log, color='blue')
    plt.plot(moving_avg, color='red')
    #plt.subplots_adjust(left=None, bottom=None, right=None, wspace=None, hspace=None)
    plt.show()
    timeseries_log_moving_avg_diff = timeseries_log - moving_avg
    timeseries_log_moving_avg_diff.dropna(inplace=True)
    print("\n\n")
    plt.subplot(212)
    #print(ts_log_moving_avg_diff)
    test_stationarity(timeseries_log_moving_avg_diff)
    plt.show()
    return moving_avg

def seperate_plots(timeseries_log):
    decomposition = seasonal_decompose(timeseries_log, freq=52)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(timeseries_log, label='Original')
    plt.legend(loc='best')
    plt.show()
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.show()
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.show()
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def ar_model(timeseries_log,timeseries_log_diff):
    #MA model:
    model = ARIMA(timeseries_log, order=(2, 1, 0))  
    results_AR = model.fit(disp=-1)  
    plt1.plot(timeseries_log_diff)
    plt1.plot(results_AR.fittedvalues, color='red')
    plt1.title('RSS: %.4f'% sum((results_AR.fittedvalues-timeseries_log_diff)**2))
    plt1.show()
    
def ma_model(timeseries_log,timeseries_log_diff):
    model = ARIMA(timeseries_log, order=(0, 1, 2))  
    results_MA = model.fit(disp=-1)  
    plt.plot(timeseries_log_diff)
    plt.plot(results_MA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-timeseries_log_diff)**2))
    plt.show()

def arima_model(timeseries_log,timeseries_log_diff):
    model = ARIMA(timeseries_log, order=(2, 1, 2))  
    results_ARIMA = model.fit(disp=-1)  
    plt.plot(timeseries_log_diff)
    plt.plot(results_ARIMA.fittedvalues, color='red')
    plt.title('RSS (Root Squared Sum): %.4f'% sum((results_ARIMA.fittedvalues-timeseries_log_diff)**2))
    plt.show()
    return results_ARIMA

def ACF_PCF_plot(timeseries_log_diff):
    lag_acf = acf(timeseries_log_diff, nlags=20)
    #print('********',lag_acf)
    lag_pacf = pacf(timeseries_log_diff, nlags=20, method='ols')
    #print('********',lag_pacf)
    
    #Plotting the ACF:
    print("Plotting the ACF:")
    plt.subplot(121)    
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    plt.show()

    #Plot PACF:
    print("Plotting the PACF:")
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    
def cumulative_sum(timeseries_log,predictions_ARIMA_diff):
    
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    print (predictions_ARIMA_diff_cumsum.head())
    #print(ts_log)
    predictions_ARIMA_log = pd.Series(timeseries_log.ix[0], index=timeseries_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
    plt.plot(timeseries_log)
    plt.plot(predictions_ARIMA_log)
    #print(predictions_ARIMA_log.head())
    return predictions_ARIMA_log

def results(timeseries,predictions_ARIMA_log):
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    #print(predictions_ARIMA)
    acc = list()
    for i in range (len(timeseries)):
        print("ACTUAL:",timeseries[i],"PREDICTED:", predictions_ARIMA[len(timeseries)-i-1])
        accu = 100 - (abs(timeseries[i] - predictions_ARIMA[len(timeseries)-i-1])/timeseries[i])*100
        acc.append(accu)
    print(sum(acc)/len(timeseries))

    plt.plot(timeseries)
    plt.plot(predictions_ARIMA)

    #value=np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts))
    #print(value)
    #acc=ts-predictions_ARIMA
    #print(acc)
    plt.title('RMSE and Predictions:')
    plt.show()
    
def main():
    #loadingdata()
    stockdata=load_edit_data()
    print("main data", stockdata)
    t_series=convert_timeseries(stockdata)
    print("tseries",t_series)
    test_stationarity(t_series)
    tseries_log=convert_stationary(t_series)
    moving_avg=data_smoothing(tseries_log)
    #Take first difference:
    timeseries_log_diff = tseries_log - tseries_log.shift()
    plt.plot(timeseries_log_diff)
    timeseries_log_diff.dropna(inplace=True)
    test_stationarity(timeseries_log_diff)
    
    seperate_plots(tseries_log)
    ACF_PCF_plot(timeseries_log_diff)
    ar_model(tseries_log,timeseries_log_diff)
    ma_model(tseries_log,timeseries_log_diff)
    results_ARIMA=arima_model(tseries_log,timeseries_log_diff)
    #calculating the difference between the consecutive ARIMA values
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy= True)
    print (predictions_ARIMA_diff.head())
    predictions_ARIMA_log=cumulative_sum(tseries_log,predictions_ARIMA_diff)
    results(t_series,predictions_ARIMA_log)
    
    
    
main()

