# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 09:58:53 2021

@author: cyn_n
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 22:02:40 2021

@author: cyn_n
"""
import streamlit as st
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
import statsmodels.api as sm
import statsmodels.api
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
from datetime import datetime
from datetime import timedelta
import yfinance as yf
import pandas as pd
import talib as ta 
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARMA 
import image
#from sklearn.ensemble import RandomForestRegrC essor
from matplotlib import pyplot

def  baja(TICK):
    
    data = yf.download(  tickers = TICK,

        period = "max",

        interval = "1d",

        
        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,

        # download pre/post regular market hours data
        # (optional, default is False)
       # prepost = True,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
    )
    return data  

def dataframes1(data):
    analisis = data['Close']
    
    suby_close= analisis.dropna()
    suby_low = data['Low'].dropna()
    suby_high = data['High'].dropna()
    suby_open = data['Open'].dropna()
    volume = data['Volume'].dropna()
    
    dataframes = pd.DataFrame(suby_close)
    dataframes.columns=['High']
    dataframes["Low"] = suby_low
    dataframes["Close"]=suby_high
    dataframes["Volume"]=volume
    dataframes["Open"]=suby_open
    dataframes.index.is_unique
    dataframes.fillna(dataframes.mean(), inplace=True)
    return dataframes 


def CCI(dataframes, ndays): 
    TP = (dataframes['High'] + dataframes['Low'] + dataframes['Close']) / 3 
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
                    name = 'CCI') 
    dataframes['CCI'] = CCI
    return dataframes

def plotcci(dataframes,n,CCI):
    NIFTY_CCI = CCI(dataframes, n)
    CCI = NIFTY_CCI['CCI'].dropna()
    
    # Plotting the Price Series chart and the Commodity Channel index below
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xticklabels([])
    plt.plot(dataframes['Close'],lw=1)
    plt.title('NSE Price Chart')
    plt.ylabel('Close Price')
    plt.grid(True)
    bx = fig.add_subplot(2, 1, 2)
    plt.plot(CCI,'k',lw=0.75,linestyle='-',label='CCI')
    plt.legend(loc=2,prop={'size':9.5})
    plt.ylabel('CCI values')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()


def fibunachi(dataframes):
    suby_close  = dataframes['Close'].dropna()    
    price_min = suby_close.min()
    price_max = suby_close.max()
    
    diff = price_max - price_min
    level1 = price_max - 0.236 * diff
    level2 = price_max - 0.382 * diff
    level3 = price_max - 0.618 * diff
    
    print( "Level", "Price")
    print( "0 ", price_max)
    print( "0.236", level1)
    print ("0.382", level2)
    print ("0.618", level3)
    print ("1 ", price_min)
    
    fig, ax = plt.subplots()
    ax.plot(suby_close, color='black')
    ax.axhspan(level1, price_min, alpha=0.4, color='lightsalmon')
    ax.axhspan(level2, level1, alpha=0.5, color='palegoldenrod')
    ax.axhspan(level3, level2, alpha=0.5, color='palegreen')
    ax.axhspan(price_max, level3, alpha=0.5, color='powderblue')
    plt.title('Retracement levels for Fibonacci ratios of 23.6%, 38.2% and 61.8%   ' + 'FIBRAPL14.MX')
    plt.ylabel("Price")
    plt.xlabel("Time")
    plt.legend("close")
    plt.show()
    
def EVM(dataframes, ndays): 
     dm = ((dataframes['High'] + dataframes['Low'])/2) - ((dataframes['High'].shift(1) + dataframes['Low'].shift(1))/2)
     br = (dataframes['Volume'] / 100000000) / ((dataframes['High'] - dataframes['Low']))
     EVM = dm / br 
     EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name = 'EVM') 
     dataframes['EVM'] = EVM_MA  
     return dataframes 
    
def plotewm(dataframes,EVM,ndays):
    EVM = EVM(dataframes, ndays)
    EVM = EVM['EVM']
    
    # Plotting the Price Series chart and the Ease Of Movement below
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xticklabels([])
    plt.plot(dataframes['Close'],lw=1)
    plt.title(' Price Chart')
    plt.ylabel('Close Price')
    plt.grid(True)
    bx = fig.add_subplot(2, 1, 2)
    plt.plot(EVM,'k',lw=0.75,linestyle='-',label='EVM(14)')
    plt.legend(loc=2,prop={'size':9})
    plt.ylabel('EVM values')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()
    
def SMA(dataframes, ndays): 
    SMA = pd.Series(dataframes['Close'].rolling(ndays).mean(), name = 'SMA') 
    dataframes['SMA']  = SMA
    return dataframes

def EWMA(data, ndays): 
     EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
     name = 'EWMA_' + str(ndays)) 
     data = data.join(EMA) 
     return data


def  plotsma(dataframes,ndays,SMA,EWMA):
    SMA_NIFTY = SMA(dataframes,ndays)
    SMA_NIFTY = SMA_NIFTY.dropna()
    SMA = SMA_NIFTY['SMA']
    ew = 200
    EWMA_NIFTY = EWMA(dataframes,ew)
    EWMA_NIFTY = EWMA_NIFTY.dropna()
    EWMA = EWMA_NIFTY['EWMA_200']
    
    # Plotting the NIFTY Price Series chart and Moving Averages below
    plt.figure(figsize=(9,5))
    plt.plot(dataframes['Close'],lw=1, label='NSE Prices')
    plt.plot(SMA,'g',lw=1, label='50-day SMA (green)')
    plt.plot(EWMA,'r', lw=1, label='200-day EWMA (red)')
    plt.legend(loc=2,prop={'size':11})
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()
    
    
def BBANDS(dataframes, nb,TICK):
    MA = dataframes[TICK].rolling(window=nb).mean()
    SD = dataframes[TICK].rolling(window=nb).std()
    dataframes['UpperBB'] = MA + (2 * SD) 
    dataframes['LowerBB'] = MA - (2 * SD)
    return dataframes

    
def plotbband(dataframes, BRANDS,nb,TICK):
    NIFTY_BBANDS = BBANDS(dataframes, nb,TICK)
    pd.concat([NIFTY_BBANDS[TICK],NIFTY_BBANDS.UpperBB,NIFTY_BBANDS.LowerBB],axis=1).plot(figsize=(9,5),grid=True)
    
def ForceIndex(dataframes, nf): 
    FI = pd.Series(dataframes['Close'].diff(nf) * dataframes['Volume'], name = 'ForceIndex') 
    dataframes['Force Index'] =  FI
    return dataframes

def pplotacf(dataframes, lag_acf,o):
  
    plot_acf(dataframes[o], lags=lag_acf)
    pyplot.show()
    
def ppltopacf(dataframes, lag_pacf,o):
    plot_pacf(dataframes[o], lags=lag_pacf)
    pyplot.show()
  
def reg_li(dataframes): 
    dataframes['t'] = range (1,len(dataframes)+1)
    
    # Computes t squared, tXD(t) and n
    dataframes['sqr t']=dataframes['t']**2
    dataframes['tXD']=dataframes['t']*dataframes['Close']
    n=len(dataframes)
    
    # Computes slope and intercept
    slope = (n*dataframes['tXD'].sum() - dataframes['t'].sum()*dataframes['Close'].sum())/(n*dataframes['sqr t'].sum() - (dataframes['t'].sum())**2)
    intercept = (dataframes['Close'].sum()*dataframes['sqr t'].sum() - dataframes['t'].sum()*dataframes['tXD'].sum())/(n*dataframes['sqr t'].sum() - (dataframes['t'].sum())**2)
    print ('The slope of the linear trend (b) is: ', slope)
    print ('The intercept (a) is: ', intercept)
    
    dataframes['forecast'] = intercept + slope*dataframes['t']
    
    # Computes the error
    dataframes['error'] = dataframes['Close'] - dataframes['forecast']
    mean_error=dataframes['error'].mean()
    print ('The mean error is: ', mean_error)
    
def dayly_returns(dataframes,option):
    list_1 = []
    
    for i in range(len(dataframes)-1):
        l1 = np.log(dataframes[option][i+1]/dataframes[option][i])
        list_1.append(l1)
    
    list_1.append('0')
    
    dataframes['trans'] = list_1[:len(dataframes)]
    dataframes['trans']= pd.to_numeric(dataframes['trans'], downcast='float')
    
    # dataframes['trans'].plot(figsize=(10,10),grid=True)
    # plt.xlabel('Time')
    # plt.ylabel('Returns')
    # plt.title(' Daily Returns')
    # plt.show()

def stochacstic_oscilator(dataframes):
    dataframes['slowk'], dataframes['slowd'] = ta.STOCH(dataframes['High'], dataframes['Low'], dataframes['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    dataframes[['slowk','slowd']].plot(figsize=(15,15))
    plt.title(' Stochastic Oscillator')
    plt.ylabel('Bounded range')    
    plt.show()
    
def average_direccional_mo(dataframes,n_adm):
    dataframes['avg'] = ta.ADX(dataframes['High'],dataframes['Low'], dataframes['Close'], timeperiod=n_adm)
    
    dataframes[['avg']].plot(figsize=(12,10))
    plt.title(' Momentum Indicator')
    plt.show()
    
def plotting_candlestick(dataframes, TICK):
    dataframes['Datee'] = dataframes.index
    dataframes['Datee'] = pd.to_datetime(dataframes['Datee'])
                                   
    set1 = {
          'x' : dataframes.Datee,
          'open'    : dataframes.Open,
          'close' : dataframes.Close,
          'high' : dataframes.High,
          'low'  : dataframes.Low,
          'type' : 'candlestick' ,   
            }  
    avg_20 = dataframes.Close.rolling(window = 20, min_periods =1).mean()
    avg_50 = dataframes.Close.rolling(window = 50, min_periods =1).mean()
    avg_200 = dataframes.Close.rolling(window = 200, min_periods =1).mean()
    
    set2 = {
          'x' : dataframes.Datee,
          'y'    : avg_20,
          'type' : 'scatter',
          'mode' : 'lines',
          'line'  : {
          'width' : 1 ,
          'color':'blue'
             },
          'name':'Moving Average of 20 periods'
          }  
    
    set3 = {
          'x' : dataframes.Datee,
          'y'    : avg_50,
          'type' : 'scatter',
          'mode' : 'lines',
          'line'  : {
          'width' : 1 ,
          'color':'yellow'
             },
          'name':'Moving Average of 50 periods'
          }  
    set4 = {
          'x' : dataframes.Datee,
          'y'    : avg_200,
          'type' : 'scatter',
          'mode' : 'lines',
          'line'  : {
          'width' : 1 ,
          'color':'black'
             },
          'name':'Moving Average of 200 periods'
          }  
    d = [set1, set2, set3, set4]
    layout =go.Layout({
        'title':{
         'text': TICK  ,
         'font': {
             'size': 25
             }
            }
        
        })
    
    fig = go.Figure(data = d, layout= layout)
    fig.show()
    
def Top_Pattern_recognition(dataframes,n_patterns,rep_ind):    

    candle_names = ta.get_function_groups()['Pattern Recognition']
    candle_names
    
    df = dataframes.copy()
    # extract OHLC 
    op = df['Open']
    hi = df['High']
    lo = df['Low']
    cl = df['Close']
    # create columns for each pattern
    l = dataframes.columns
    cols = [i for i in l if i != 'Open' and  i != 'Close' and  i != 'High' and  i != 'Low'  ]
    df.drop(cols,axis = 1, inplace = True)
    for candle in candle_names:
        
    # below is same as;
    # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
        df[candle] = getattr(ta, candle)(op, hi, lo, cl)
        
    # =============================================================================
    # Ranking the patterns
    # We successfully extracted candlestick patterns using TA-Lib. With few lines of code, we can condense
    # this sparse information into a single column with pattern labels. But first, we need to handle the 
    # cases where multiple patterns are found for a given candle. To do that, we need a performance metric 
    # to compare patterns. We will use the “Overall performance rank” from the patternsite.    
    # =============================================================================
    
    cols = [i for i in df.columns if i != 'Open' and  i != 'Close' and  i != 'High' and  i != 'Low'  ]
    df_abs = df.copy()
    df_ = df_abs.abs()
    df_["sum"] = df_.sum(axis=1)
    df['sum'] = df_['sum']
    d = df[df['sum']>100*rep_ind]
    d1 = d.sort_values(by='sum', ascending=False)[n_patterns:]
    d1 = d1.drop(d1.columns[d1.eq(0).all()], axis=1)
    return d1 


def imprimir_mejores(d1,TICK):
    d1.columns
    count = 0
    count1 = 1
    i = 'CDL3BLACKCROWS'
    for i in d1.columns:
        count1  = 1 + count1
        if count1 != (len(d1.columns)-1):
            if i == 'CDL3BLACKCROWS':
                count +=  1
            
                print(i +': ''Overall performance is outstanding. '
                      
                      'Candles in a downward price trend will qualify. The pattern acts as a bearish reversal of the upward price trend and the overall performance is outstanding.'
            
            'A check of the performance over 10 days shows some startling results: Do not trade this if the breakout is downward. Only upward breakouts are worth considering. If you remove the 10 day restriction, then the worst performance comes from three black crows in a bull market, regardless of the breakout direction. \n')
            if i == 'CDL3INSIDE':
                count +=  1
            
                print(i +': ''Rare and Highly reliable'  \
                      'Rhe pattern acts as a bearish reversal 65% of the time. The reason for the comparatively high reversal rate is because price closes near the bottom of the candlestick pattern and all a reversal has to do it post a close below the bottom of the candle pattern. That is much easier to do than close above the other end (the top). \n')
                    
            if i == 'CDL3OUTSIDE':
                count +=  1
            
                print(i +': '' Frequent bullish reversal but very works well'  \
                      'The three outside up candlestick acts as a bullish reversal both in theory and in reality. And it does so quite well. It has a high frequency number, so you should be able to find is as often as feathers on a duck. The overall performance is also quite good and that means the price trend, post breakout, is worthwhile if not downright tasty. However, you will want to avoid this candlestick pattern if you hold it for a short term (10 days). Upward breakouts under those conditions do particularly lousy. For longer term holds, avoid those in a bull market after a downward breakout. \n')
                    
            if i == 'CDL3WHITESOLDIERS':
                count +=  1
            
                print(i +': '' Bullish reversal 82% of the time,'  \
                      ' It acts as a bullish reversal 82% of the time, ranking 3 out of 103 candlestick types, where 1 is best. or a downward breakout to occur, price would have to make a serious drop and plummet even more to push up the performance score. In fact, the high performance is due to just those factors: few candles with downward breakouts.  \n') 
            if i == 'CDLENGULFING':
                count +=  1
            
                print(i +': ''Common'  \
                      'A closer look at the numbers shows that downward breakouts are where this pattern outperforms. The best move 10 days after an upward breakout is a drop of 1.18%. Usually you would see a rise 10 days after an upward breakout but not in this candlestick. Thus, if you are going to rely on this candlestick then look for a downward breakout. The best move appears in a bear market, so that is the way to trade this one. \n') 
            
            if i == 'Respectable':
                count +=  1
            
                print(i +': ''The evening doji star is one of the better performing candlestick patterns. It has a high reversal rate, ranking 12th, and the performance over time is respectable, too, but not outstanding. A check of the numbers shows that downward breakouts are weakest, and upward ones are strongest with the bear market/up breakout configuration  doing the best of the bunch.  \n'  \
                      '') 
            
            if i == 'CDLEVENINGSTAR':
                count +=  1
            
                print(i +': ''Highly reliable as market reversal'  \
                      'The evening star acts as a bearish reversal of the upward price trend, and when the breakout occurs, hold on for dear life. The overall performance rank is very high, but most of the performance is due to upward breakouts, not downward ones. In fact, you will want to avoid trading the evening star after a downward breakout in a bull market. \n') 
            
            if i == 'CDLHAMMER':
                count +=  1
            
                print(i +': ''Highly respected'  \
                      'If you project the height of the candle in the direction of the breakout (candle top for upward breakouts and candle bottom for downward ones), price meets the target 88% of the time, which is very good. The best average move occurs after a downward breakout in a bear market. Price drops an average of 4.12% after a hammer, placing the rank at 48 where 1 is best. That, of course, is just mid range out of the 103 candle types studied. A good performance would be a move of 6% or more. \n')
                    
            
            if i == 'CDLHOMINGPIGEON':
                count +=  1
            
                print(i +': ''Realiable'  \
                      'The homing pigeon is supposed to act as a bullish reversal, '
                    '  but testing found that it performs as a bearish continuation 56% of the time. That is what I call "near random" because you won''t'' be able to tell the breakout direction with any certainty. Despite the poor reversal rate, the overall performance is quite good. \n') 
            
            if i == 'CDLINVERTEDHAMMER':
                count +=  1
            
                print(i +': ''Reliable for bear continuation'  \
                      'A hammer is a single candle line in a downtrend, but an inverted hammer is a two line candle, also in a downtrend. The inverted hammer is supposed to be a bullish reversal candlestick, but it really acts as a bearish continuation 65% of the time. The overall performance ranks it 6 out of 103 candles, meaning the trend after the candle often results in a good sized move. \n') 
            
            if i == 'CDLLONGLINE':
                count +=  1
            
                print(i +': ''Reliable'  \
                      'Above the stomach is a simple candlestick pattern and yet it works very well, functioning as a bullish reversal 66% of the time.  \n') 
            
            if i == 'CDLMORNINGDOJISTAR':
                count +=  1
            
                print(i +': ''Reliable'  \
                      'Taking a microscope to the performance numbers shows that downward breakouts outperform upward ones both in the 10-day performance rank and by the average move 10 days after the breakout. If you remove the 10-day restriction and just measure the move to the trend end, you will find that downward breakouts in a bull market are the weak performers, but in a bear market, they are the strongest performers. \n') 
            
            if i == 'CDLMORNINGSTAR':
                count +=  1
            
                print(i +': ''Not frequent but highly reliable'  \
                      'If you arbitrarily sell 10 days after the breakout, you will find that the morning star after an upward breakout is the weakest performer. However, just letting the trend end when it ends instead of imposing a time limit shows that upward breakouts have better post-breakout performance than downward ones. That tells me the trend after the breakout from a morning star takes a while to get going but it tends to keep moving up. Patience is probably a good word for what you need when trading this candle pattern. \n') 
            
            if i == 'CDLPIERCING':
                count +=  1
            
                print(i +'Higher priority but not as frequent'  
                      'Overall performance is good, too, suggesting the price trend after the breakout is a lasting and profitable one. The piercing pattern does best in a bear market, especially after a downward breakout. Upward breakouts in a bull market are the weakest of the four combinations of bull/bear market and up/down breakout direction. \n') 
            
            if i == 'CDLRISEFALL3METHODS':
                count +=  1
            
                print(i +': ''Highly reliable bearish continuation but rare' \
                      'The falling three methods pattern acts as a bearish continuation 71% of the time, giving it a rank of 7 which is very high (1 is best). As I mentioned, it is a rare bird, having a frequency rank of 91 out of 103 candles where 1 means it occurs most often.  \n') 
            
            if i == 'CDLSEPARATINGLINES':
                count +=  1
            
                print(i +': ''Reliable for bullish continuation'  \
                      'he bullish separating lines candle pattern acts in theory as it does in reality, that of a bullish continuation 72% of the time. That ranks 6th of out 103 candles, which is very strong. The frequency rank is 76, so this candle pattern will not be easy to spot because it is rare.  \n') 
            
            if i == 'CDLSPINNINGTOP':
                count +=  1
            
                print(i +': ''Most frequence'  \
                      'As I mentioned in the introduction, the black spinning top works in theory as it does in reality: indecision. Price breaks out in any direction almost equally. With a frequency rank of 1, the candle is prolific, so you will see it often in a historical price series. The best move 10 days after a breakout occurs when price drops 3.36% in a bear market.  \n') 
        else:
            if count == 0:
                
                print( 'There was not a frequent or reliable candlestick pattern for: ' + TICK + ' considering the parttern available in the ta-lib module. \n'  ) 
        


def graficar_predicciones(real, prediccion,option):
    plt.plot(real[0:len(prediccion)],color='red', label='Valor real de la acción')
    plt.plot(prediccion, color='blue', label='Predicción de la acción')
    plt.ylim(1.1 * np.min(prediccion)/2, 1.1 * np.max(prediccion))
    plt.xlabel('Tiempo')
    plt.ylabel('Valor de la acción')
    plt.title(' Predicciones precio  '+ option)
    plt.legend()
    plt.show()

#
# Lectura de los datos
#
#dataset = pd.read_csv('AAPL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
#dataset.head()

#
# Sets de entrenamiento y validación 
# La LSTM se entrenará con datos de 2016 hacia atrás. La validación se hará con datos de 2017 en adelante.
# En ambos casos sólo se usará el valor más alto de la acción para cada día
#
#set_entrenamiento = [dataframes['High'][:int(round(len(dataframes)*.8,0))],dataframes['Datee'][:int(round(len(dataframes)*.8,0))]]
def lstm_prediccion(graficar_predicciones,dataframes,option, na, layer):
    if option == 'Low':
        set_entrenamiento = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,2:3]
        set_validacion = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,2:3]    
    
    
    elif option == 'High':
        set_entrenamiento = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,1:2]
        set_validacion = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,1:2]
    
    elif option == 'Open':
        set_entrenamiento = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,0:1]
        set_validacion = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,0:1]
    
    
    elif option == 'Close':  
        set_entrenamiento = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,3:4]
        set_validacion = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,3:4]
    
        
    dataframes['Datee'] = pd.to_datetime(dataframes['Datee'])
    a = dataframes['Datee'][1]
    a = a.year
    a1 = dataframes['Datee'][int(round(len(dataframes)*.8,0))]
    a1 = a1.year
    b1 = a1+1
    b2 = dataframes['Datee'][len(dataframes)-1].year
    
    set_entrenamiento[option].plot(legend=True)
    set_validacion[option].plot(legend=True)
    
    plt.legend( [f'Entrenamiento {a} a {a1}', f'Validación {b1} a {b2}'])
    plt.title(' Datos de entrenamiento y validación.')
    plt.show()
    
    # Normalización del set de entrenamiento
    sc = MinMaxScaler(feature_range=(0,1))
    set_entrenamiento_escalado = sc.fit_transform(set_entrenamiento)
    
    # La red LSTM tendrá como entrada "time_step" datos consecutivos, y como salida 1 dato (la predicción a
    # partir de esos "time_step" datos). Se conformará de esta forma el set de entrenamiento
    time_step = 60
    X_train = []
    Y_train = []
    m = len(set_entrenamiento_escalado)
    
    for i in range(time_step,m):
        # X: bloques de "time_step" datos: 0-time_step, 1-time_step+1, 2-time_step+2, etc
        X_train.append(set_entrenamiento_escalado[i-time_step:i,0])
    
        # Y: el siguiente dato
        Y_train.append(set_entrenamiento_escalado[i,0])
        X_train, Y_train = np.array(X_train), np.array(Y_train)
    
    # Reshape X_train para que se ajuste al modelo en Keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    #
    # Red LSTM
    #
    dim_entrada = (X_train.shape[1],1)
    dim_salida = 1
    #son 50 neuronas a usar 
    #son 50 neuronas a usar 
    modelo = Sequential()
    #fin de agregar capas
    #modelo.add(Dense(layer))
        # modelo.add(Dense(12,input_dim = 8,activation = 'relu'))
        # modelo.add(8,activation = 'relu')
        # modelo.add(Dense(1, activation= 'sigmoid'))
    
    modelo.add(LSTM(units=na, input_shape=dim_entrada))
    #entrada, lo cual es 60 datos
    modelo.add(Dense(units=dim_salida))
    #rmsprop funciona de manera similar al gradiente desciendinte
    modelo.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
    #salida
    modelo.fit(X_train,Y_train,epochs=20,batch_size=32,verbose=0)
    #lotes de 32 ejemplos y un total de 20 iteraciones
    
    #
    # Validación (predicción del valor de las acciones)
    #
    x_test = set_validacion.values 
    # x_test = sc.transform(x_test)
    x_test = sc.transform(x_test)
    
    X_test = []
    for i in range(time_step,len(x_test)):
        X_test.append(x_test[i-time_step:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    
    prediccion = modelo.predict(X_test)
    prediccion = sc.inverse_transform(prediccion)
    df = pd.DataFrame(prediccion)
    # Graficar resultados
    if df.isnull().sum()[0] == df.shape[0]:
       print('No se pudo ajustar el modelo LSTM  a los datos historicos')
    else:
        graficar_predicciones(set_validacion.values,prediccion,option)
        
    
        #Acuracy de modelo square(y_true - y_pred)
        mse = tf.keras.losses.MeanSquaredError()
        l1 = mse(set_validacion[:len(prediccion)], prediccion).numpy()
        print(f'Mean Squared Error is:{l1} \n')
        # in each value x in y_true and y_pred.
        # loss = 0.5 * x^2                  if |x| <= d
        # loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
        
        h = tf.keras.losses.Huber()
        l = h(set_validacion[:len(prediccion)],prediccion).numpy()
        print(f'Humbert Losses is:{l} ')
        
    
#Sarima movelo Tim er series
def destranform_returns(dataframes,option,predictions):
    list_1 = []
    
    for i in range(len(dataframes)-1):
        l1 = np.exp(predictions[i:i+1])*dataframes[option][i]
        list_1.append(list(l1)[0])
    l = list_1[:1]
    list_1.append('0')
    
    dataframes['pre_trans'] = list_1[:len(dataframes)]
    dataframes['trans']= pd.to_numeric(dataframes['trans'], downcast='float')

def def_seasonal(option, dataframes,TICK,freq_des):
    dataframes = dataframes.dropna()    
    
    if option == 'Low':
            train = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,2:3]
            test = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,2:3]    
        
      
    elif option == 'High':
        train = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,1:2]
        test = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,1:2]
    
    elif option == 'Open':
        train = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,0:1]
        test = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,0:1]
    
    
    elif option == 'Close':  
        train = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,3:4]
        test = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,3:4]
    
    try:
       descomposicion = sm.tsa.seasonal_decompose(dataframes[option],model='additive', freq=freq_des)  
       fig = descomposicion.plot()

       plt.savefig('./imag/descomposicional.png') 
    except:
      print("Try with another freq for the descomposicion ")

    
def destranform_returns(dataframes,option,predictions):
    list_1 = []
    
    for i in range(len(dataframes)-1):
        l1 = np.exp(predictions[i:i+1])*dataframes[option][i]
        list_1.append(list(l1)[0])
    l = list_1[:1]
    list_1.append(list_1[-1])
    
    dataframes['pre_trans'] = list_1[:len(dataframes)]
    dataframes['trans']= pd.to_numeric(dataframes['trans'], downcast='float')

def def_seasonal(option, dataframes,TICK,freq_des):
    dataframes = dataframes.dropna()    
    
    if option == 'Low':
            train = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,2:3]
            test = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,2:3]    
        
      
    elif option == 'High':
        
        train = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,1:2]
        test = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,1:2]
    
    elif option == 'Open':
        train = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,0:1]
        test = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,0:1]
    
    
    elif option == 'Close':  
        train = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,3:4]
        test = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,3:4]
    
    try:
       descomposicion = sm.tsa.seasonal_decompose(dataframes[option],model='additive', freq=freq_des)  
       fig = descomposicion.plot()

       plt.savefig('./imag/des.png')
    except:
      print("Try with another freq for the descomposicion ")
      
      print('La series de tiempo no estacionaria ')
       
    return train,test  
        #Daily returns
def arma1_fortrans(adfuller,dataframes):
         
        #Volvemos aplicar el test    
    try:
       test=adfuller(abs(dataframes['trans']))[1]
       if test > .05:
         U =    'La series de tiempo no estacionaria '
       else:
          U  =  'The transformation of the time series is stationary, we will proceed to the application of the ARMA model'

    except:
          U=      "For the nature of the data it is not possible to predict these series with ARMA"
     
    return U


def plot_lags(lag_acf,lag_pacf,dataframes):      
  #autocorrelation function (acf) 
             o = 'trans'
             
             try:
                m11 = pplotacf(dataframes, lag_acf,o)
                plt.savefig('./imag/acf.png')
             #autocorrelation function (PACF) removes the effect of shorter lag autocorrelation 
                m1 = ppltopacf(dataframes, lag_pacf,o)
                plt.savefig('./imag/pacf.png')
                 
             except:
                 print("Please choose another another lower lag, with minor value.")
             return m1, m11
        
def modelo_altranformarlo(dayly_returns,pplotacf,dataframes,option,plot_lags,lag_acf,lag_pacf,p,q): 
    #aqui haceos la transgtomacion
    dayly_returns(dataframes,option)
    o = option
        
    try:
       
        plot_lags(lag_acf,lag_pacf,dataframes)
         
         #ahora que ya le dimos la herramiento al usuario de obtener las lags le pedimos que las ingrese
             
        model = ARMA(dataframes['trans'], order=(p, q))
        model_fit = model.fit(disp=False)
        # make prediction
        #predictions = model_fit.predict(len(sensor['userAcceleration.x'])-3, len(sensor['userAcceleration.x'])-1)
        predictions = model_fit.predict(start=len(dataframes['trans']), end=len(dataframes['trans'])+len(dataframes['trans'])-1, dynamic=False)    
        #destranformamos 
        destranform_returns(dataframes,option,predictions)
        prediction  = pd.DataFrame(predictions)
        
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(2, 1, 1)
        ax.set_xticklabels([])
        plt.plot(dataframes[option],lw=1)
        plt.title('ARMA Price Chart')
        plt.ylabel(f'{option}  Price')
        plt.grid(True)
        bx = fig.add_subplot(2, 1, 2)
        plt.plot(dataframes['pre_trans'],'k',lw=0.75,linestyle='-',label='ARIMA')
        plt.legend(loc=2,prop={'size':9.5})
        plt.ylabel('ARMA values')
        plt.grid(True)
        plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.show()
        plt.savefig('./imag/ARMA.png')
        
                
        prediction1  = pd.DataFrame(dataframes['pre_trans'])
                                     
        #the unexplained variance, and has the useful property of being in the same units as the response variable. Lower values of RMSE indicate better fit.
        m = statsmodels.tools.eval_measures.rmse(dataframes[option], prediction1['pre_trans'].astype(int))
        
        print(f'The RMSE, is the unexplained variance and is: {m} \n')
        predictions = model_fit.predict(start=len(dataframes[o]), end=len(dataframes[o])+len(dataframes[o])-1, dynamic=False)    
        m1 = statsmodels.tools.eval_measures.rmse(dataframes[option], prediction[0].astype(int))
        print(f'Without the transformation of the data the RMSE, is the unexplained variance and is: {m1} ')
    except:
             print("For the nature of the data it is not possible to predict these series with ARMA")
  
    return m
def arma_sintrans(dataframes,lag_acf,lag_pacf,ARMA,option,p,q):
    
    print('La serie de tiempo es estacionaria procederemos a la aplicaciòn del modelo ARMA')
    o = option
    
    plot_lags(lag_acf,lag_pacf,dataframes)
        
    
    #ahora que ya le dimos la herramiento al usuario de obtener las lags le pedimos que las ingrese
        
    model = ARMA(dataframes[o], order=(p, q))
    model_fit = model.fit(disp=False)
    # make prediction
    #predictions = model_fit.predict(len(sensor['userAcceleration.x'])-3, len(sensor['userAcceleration.x'])-1)
    predictions = model_fit.predict(start=len(dataframes[o]), end=len(dataframes[o])+len(dataframes[o])-1, dynamic=False)    
    #destranformamos 
    prediction  = pd.DataFrame(predictions)
    
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xticklabels([])
    plt.plot(dataframes[option],lw=1)
    plt.title('ARMA Price Chart')
    plt.ylabel(f'{option}  Price')
    plt.grid(True)
    bx = fig.add_subplot(2, 1, 2)
    plt.plot(prediction,'k',lw=0.75,linestyle='-',label='ARIMA')
    plt.legend(loc=2,prop={'size':9.5})
    plt.ylabel('ARMA values')
    plt.grid(True)
    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.show()
    plt.savefig('./imag/ARMA.png')
            
    #the unexplained variance, and has the useful property of being in the same units as the response variable. Lower values of RMSE indicate better fit.
    m = statsmodels.tools.eval_measures.rmse(dataframes[option], prediction[0].astype(int))
    
    print(f'The RMSE, is the unexplained variance and is: {m} ')
    return m 
    
def arma_fit(option,dataframes,TICK,dayly_returns,pplotacf,ppltopacf,destranform_returns,freq_des,lag_acf,lag_pacf,def_seasonal,p,q,arma_sintrans):
      #ploteamos la descomposicion seasonal
    
        #primero se ve si los datos son estacionarios sin formula
        try:
            print(f'p value {TICK} Price: ', adfuller(abs(dataframes[option]))[1])
            test=adfuller(dataframes[option])[1]
        except:     
             print("For the nature of the data it is not possible to predict these series with ARMA")
         
     #se grafican las laf  
        lag_plot(dataframes[option]);
          
        
        if test > .05:
            
        
        #Grid search
            modelo_altranformarlo(dayly_returns,pplotacf,dataframes,option,plot_lags,lag_acf,lag_pacf,p,q)
        
        
        else:
             
           arma_sintrans(dataframes,lag_acf,lag_pacf,ARMA,option,dataframes,p,q)
            