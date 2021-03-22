# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 17:24:59 2021

@author: cyn_n
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_pacf
from datetime import datetime
from datetime import timedelta
import statsmodels.api as sm
import statsmodels.api
from PIL import Image
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA

import functions_ss as f
from functions_ss import CCI
from functions_ss import EVM
from functions_ss import EWMA,plot_lags
from functions_ss import SMA,arma1_fortrans
from functions_ss import SMA,dayly_returns,pplotacf,ppltopacf,destranform_returns
from functions_ss import BBANDS, ForceIndex,graficar_predicciones,destranform_returns
import talib as ta 
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
import requests
from compress import Compressor

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf


import warnings
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
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
import streamlit as st
import time
import image 
import base64
from PIL import Image
from functions_ss import SMA,dayly_returns,pplotacf,destranform_returns,ppltopacf
from functions_ss import BBANDS, ForceIndex,graficar_predicciones,destranform_returns,def_seasonal

#funciones


#poner la imagen en el backgroung
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def SMA(dataframes, ndays): 
    SMA = pd.Series(dataframes['Close'].rolling(ndays).mean(), name = 'SMA') 
    dataframes['SMA']  = SMA
    return dataframes

def EWMA(data, ndays): 
     EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
     name = 'EWMA_' + str(ndays)) 
     data = data.join(EMA) 
     return data


set_png_as_page_bg('./imag/C.PNG')

def plo_lag(option,dataframes):
    pd.plotting.lag_plot(dataframes[option], lag=1)
    plt.savefig('./imag/Captura1.PNG')
          


#aqui empieza
st.markdown( """ <style> .sidebar .sidebar-content { background-image: linear-gradient($gray-900); color: red; } </style> """, unsafe_allow_html=True, )

nav = st.sidebar.selectbox("Menu",['Home', "Prediction Time Series", "Prediction Neural Network", 'Statistics Indicators', "Candlestick pattern recognition"])

unique = ["FIBRAPL14.MX","GNP.MX", "ALPEKA.MX", "GMEXICOB.MX", "PE&OLES.MX", "BIMBOA.MX", "GRUMAB.MX","SITESB-1.MX" ,"CADUA.MX"]
Selected_stock = st.sidebar.selectbox("Stock",unique)

# if nav == "Home":
    
    
   
#           
                      
if nav == "Home":
    
    
   	col1 , col2 = st.beta_columns([4,1])
   	col2.subheader('')
   	col2.subheader('')
   	col2.subheader('')
   	col2.subheader('')
   	
   	col2.image('./imag/Captura.PNG',width = 500)
   	col2.subheader('')
   	col1.subheader('')
   
   	col1.subheader('Presentación')
   	col1.markdown('')
   	col1.markdown('')
   	col1.subheader('Stock Prediction')
   	col1.markdown('')
   	col1.markdown('This app gives you the prediction with Neural Networks and Time Series. It also gives you important financial indicators and an explicit explication of the candlestick pattern of each stock! ')
   	col1.markdown('This app is interacts with the user to predict with Two models Autoregressive Moving Average (ARMA) and Long Short-Term Memory Networks (LSTM).')
   	col1.markdown('It gives you indicators which demonstrate de trend of the stock' )
   	col1.markdown('The app has Candlestick Pattern Recognition')
   	col1.subheader('Structure:')
   	col1.markdown('')
       
   	col1.subheader('1. Prediction Time Series')
   	col1.markdown('')
   	
   	col1.subheader('2. Prediction Neural Network ')
   	col1.markdown('')
   	
   	col1.subheader('3. Statistics Indicators')
   	col1.markdown('' )
   	col1.subheader('4. Candlestick pattern recognition')
       

if nav == "Prediction Time Series":
    st.header('Time series (ARMA)')
    
    st.markdown("""
    The ARMA (p, q) processes is a mixed model that has an autoregressive part and a moving average part, where p is the order of the autoregressive (AR) part and q is the order of the moving average (MA)!
    * First, we analyze the data and apply the corresponding test
    * After having the proper condition of the data to me a good fit for ARMA 
    * We let the user decide the order of ARMA (p,q), we give the user the tools to make an assertive based on facts. 
    * Finally we compare the predictions with our original data. 
    """)

    TICK = Selected_stock
    data  =  f.baja(TICK)
    dataframes  =   f.dataframes1(data) 
    
# # reframe as supervised learning
#     p =1
#     q =1
# #for f.mx el primero quefrecuencia = 30
#     lag_acf = 20 
#     lag_pacf = 20
#     freq_des = 30
#     f.arma_fit(option,dataframes,TICK,dayly_returns,pplotacf,ppltopacf,destranform_returns,freq_des,lag_acf,lag_pacf)
    global option 
    option = st.radio( "Please choose the data yo want to predict :",('Close', 'Open', 'High','Low'))
   # if option == 'Low':
    print(option)
    #def_seasonal(option, dataframes,TICK,freq_des)
    st.write("We will make prediton with", option, 'history prices of', TICK)
    freq_des = st.slider('Would you choose frecuency to descompose de seasonal chart', 0, 10, 50)
    
    img1 = def_seasonal(option, dataframes,TICK,freq_des)
    plt.savefig('./imag/descomposicional.png')
    image1 = Image.open('./imag/descomposicional.png')
    st.image(image1, caption='Seasonal Descompotional')

    try:
        
          st.write('We will test if the data is stationary with the adfuller test') 
          st.write('P-Value ',TICK , 'Price: ', adfuller(abs(dataframes[option]))[1]) 
          test=adfuller(dataframes[option])[1]
          #lag_plot(dataframes[option])
          # # im = 
          plo_lag(option,dataframes)
          #st.image(Image.open('Captura1.png'), caption='Lag plot')
          plt.savefig('./imag/laa.png')
          st.write(' A “lag” is a fixed amount of passing time; One set of observations in a time series is plotted (lagged) against a second, later set of data. The kth lag is the time period that happened “k” time points before time i.')
          st.image(Image.open('./imag/laa.png'), caption='Lag plot')
          st.write('We will transform our Data because it is not stacionary, we will apply de following transformation:')
          st.image(Image.open('./imag/trans.png'), caption='Continuous returns')
          st.write('After the tranformation we will apply de adfuller test again.')
          dayly_returns(dataframes,option)
          st.write('P-Value ',TICK , 'Price: ', adfuller(abs(dataframes['trans']))[1])
          U = arma1_fortrans(adfuller,dataframes)
          st.write(U)
          
          st.write('Based on the following plots choose the p and q for the ARMA model')
          lag_plot(dataframes[option])
          lag_acf = st.slider('Would you choose  the lags for the autocorrlation ', 0, 10, 50)
          lag_pacf = st.slider('Would you choose  the lags for the partial autocallation', 0, 10, 50)
          
          #plot_lags(lag_acf,lag_pacf,dataframes)
          o = 'trans'
           
          try:
              m11 = pplotacf(dataframes, lag_acf,o)
              plt.savefig('./imag/acf.png')
              image11 = Image.open('./imag/acf.png')
              st.image(image11, caption='')
      
            #autocorrelation function (PACF) removes the effect of shorter lag autocorrelation 
              m1 = ppltopacf(dataframes, lag_pacf,o)
              plt.savefig('./imag/pacf.png')
              image12 = Image.open('./imag/pacf.png')
              st.image(image12, caption='')
       
          except:
               st.write("Please choose another another lower lag, with minor value.")
          
          p = st.slider('Choose de order of p ?', 0, 50, 1)
          q= st.slider('Choose de order of q ?', 0, 50, 1)
           
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
          image13 = Image.open('./imag/ARMA.png')
          st.image(image13, caption='')
       
                
          prediction1  = pd.DataFrame(dataframes['pre_trans'])
                                     
        #the unexplained variance, and has the useful property of being in the same units as the response variable. Lower values of RMSE indicate better fit.
          m = statsmodels.tools.eval_measures.rmse(dataframes[option], prediction1['pre_trans'].astype(int))
        
          st.write('The RMSE, is the unexplained variance and is', m)
          predictions = model_fit.predict(start=len(dataframes[o]), end=len(dataframes[o])+len(dataframes[o])-1, dynamic=False)    
          m1 = statsmodels.tools.eval_measures.rmse(dataframes[option], prediction[0].astype(int))
          st.write('Without the transformation of the data the RMSE, is the unexplained variance and is:' ,m1 )
    
        #  st.image('laa.png', caption='Seasonal Descompotional')
        
    except:     
            st.markdown("For the nature of the data it is not possible to predict these series with ARMA")


    
if nav == "Prediction Neural Network":
    
    st.header('Neural Networks')
    
    st.markdown("""
    Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior.            
    A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.
    * First, we divide the training and the test  set. 
    * After we normilize the training set.
    *The LSTM will have time_step = 60 data entrance (consecutive) and a data output
    *We let the user chose the layer and the neurons and the method to optimize.
    * Finally we compare the predictions with our original data. 
    """)

    TICK = Selected_stock
    data  =  f.baja(TICK)
    dataframes  =   f.dataframes1(data) 
    option = st.radio( "Please choose the data yo want to predict :",('Close', 'Open', 'High','Low'))
    print(option)
    #def_seasonal(option, dataframes,TICK,freq_des)
    st.write("We will make prediton with", option, 'history prices of', TICK)
          
    try:
        
        if option == 'Low':
            set_entrenamiento = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,1:2]
            set_validacion = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,1:2]    
        
    
        elif option == 'High':
            set_entrenamiento = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,0:1]
            set_validacion = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,0:1]
        
        elif option == 'Open':
            set_entrenamiento = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,0:1]
            set_validacion = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,0:1]
        
        
        elif option == 'Close':  
            set_entrenamiento = dataframes[:int(round(len(dataframes)*.8,0))].iloc[:,2:3]
            set_validacion = dataframes[int(round(len(dataframes)*.8,0)):].iloc[:,2:3]
        
        dataframes['Datee'] = dataframes.index
        dataframes['Datee'] = pd.to_datetime(dataframes['Datee'])
        a = dataframes['Datee'][1]
        a = a.year
        a1 = dataframes['Datee'][int(round(len(dataframes)*.8,0))]
        a1 = a1.year
        b1 = a1+1
        b2 = dataframes['Datee'][len(dataframes)-1].year
        
        fig = plt.figure(figsize=(7,5))
        set_entrenamiento[option].plot(legend=True)
        set_validacion[option].plot(legend=True)
    
        plt.legend( [f'Entrenamiento {a} a {a1}', f'Validación {b1} a {b2}'])
        plt.title(' Datos de entrenamiento y validación.')
        plt.savefig('./imag/TRAIN.png')
        image14 = Image.open('./imag/TRAIN.png')
        st.image(image14)
        
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
        na = st.slider('Would you choose the total number of neurons to use in de model', 0, 100, 20)
        layer = st.slider('Would you choose the number of total layers', 0, 5, 1)
     
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
        op = st.sidebar.selectbox("Choose the optimizers",['RMSprop', "Adam", "Adadelta", 'Adagrad', "SGD",'Adamax','Nadam','Ftrl'])

        
        modelo.compile(optimizer=op, loss='mse', metrics=['accuracy'])
        #salida
        modelo.fit(X_train,Y_train,epochs=20,batch_size=32,verbose=0)
        #lotes de 32 ejemplos y un total de 20 iteraciones
        
        #
        # Validación (predicción del valor de las acciones)
        #
        x_test = set_validacion.values 
        x_test = sc.transform(x_test)
        
        X_test = []
        for i in range(time_step,len(x_test)):
            X_test.append(x_test[i-time_step:i,0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        
        prediccion = modelo.predict(X_test)
        prediccion = sc.inverse_transform(prediccion)
        df = pd.DataFrame(prediccion[:,0])
        
        # plt.xlabel('Tiempo')
        # plt.ylabel('Valor de la acción')
        # plt.title(' Predicciones precio  '+ option)
        # plt.legend()
        # plt.plot(set_validacion.values[0:len(prediccion)],color='red', label='Valor real de la acción')
        # plt.plot(prediccion, color='blue', label='Predicción de la acción')
        # plt.savefig('neuronal.png')
        # image5 = Image.open('neuronal.png')
    # st.image(image5)

        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(2, 1, 1)
        ax.set_xticklabels([])
        plt.plot(set_validacion.values[0:len(prediccion)],lw=1)
        plt.title(' Price Chart')
        plt.ylabel(f'{option}  Price')
        plt.grid(True)
        bx = fig.add_subplot(2, 1, 2)
        plt.plot(prediccion,'k',lw=0.75,linestyle='-',label='LSTM')
        plt.legend(loc=2,prop={'size':9.5})
        plt.ylabel('LSTM values')
        plt.grid(True)
        plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.show()
        plt.savefig('./imag/neuronal.png')
        image15 = Image.open('./imag/neuronal.png')
        st.image(image15, caption='')
   


        mse = tf.keras.losses.MeanSquaredError()
        l1 = mse(set_validacion[:len(prediccion)], prediccion).numpy()
        st.write('Mean Squared Error is:',l1)
        # in each value x in y_true and y_pred.
        # loss = 0.5 * x^2                  if |x| <= d
        # loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
        
        h = tf.keras.losses.Huber()
        l = h(set_validacion[:len(prediccion)],prediccion).numpy()
        st.write('Humbert Losses is:' ,l)
        
     
    except:     
        st.markdown("For the nature of the data it is not possible to predict these series with Neuronal Network")

if nav == "Candlestick pattern recognition":
    
    st.header('Candlestick pattern recognition')
    
    st.markdown("""
    Candlestick charts are a technical tool that packs data for multiple time frames into single price bars. We look for bearish o bullish pattern in order to predict based in our past. 
    
     
    """)

    TICK = Selected_stock
    data  =  f.baja(TICK)
    dataframes  =   f.dataframes1(data) 
    option = st.radio( "Please choose the data yo want to predict :",('Close', 'Open', 'High','Low'))
    print(option)
    #def_seasonal(option, dataframes,TICK,freq_des)
    st.write("We will make prediction with", option, 'history prices of', TICK)
    
    try:
        
        dataframes['Datee'] = dataframes.index
        
        # m  =  list( dataframes['Datee'].values.astype('datetime64[D]')  )
        # dataframes['Datee'] = pd.DataFrame([x for x in m])
        dataframes['Datee']   = pd.to_datetime(dataframes['Datee'],format = '%d-%m-%Y',errors  ='raise')
        df = dataframes
            
        avg_20 = df.Close.rolling(window=20, min_periods=1).mean()
        avg_50 = df.Close.rolling(window=50, min_periods=1).mean()
        avg_200 = df.Close.rolling(window=200, min_periods=1).mean()
        set1 = { 'x': df.Datee,  'close': df.Close, 'high': df.High, 'low': df.Low, 'open': df.Open, 'type': 'candlestick',}
        set2 = { 'x': df.Datee, 'y': avg_20, 'type': 'scatter', 'mode': 'lines', 'line': { 'width': 1, 'color': 'blue' },'name': 'MA 20 periods'}
        set3 = { 'x': df.Datee, 'y': avg_50, 'type': 'scatter', 'mode': 'lines', 'line': { 'width': 1, 'color': 'yellow' },'name': 'MA 50 periods'}
        set4 = { 'x': df.Datee, 'y': avg_200, 'type': 'scatter', 'mode': 'lines', 'line': { 'width': 1, 'color': 'black' },'name': 'MA 200 periods'}
        data = [set1, set2, set3, set4]
        layout =go.Layout({'title':{  'text': TICK  ,  'font': {      'size': 25      }     } })
        fig = go.Figure(data=data,layout= layout)
       
        fig.update_layout(height=700)
        st.plotly_chart(fig,use_container_width=True)
        #Patern Recognition
        n_patterns = 10
        rep_ind = 1
        d1 =f.Top_Pattern_recognition(dataframes,n_patterns,rep_ind)
        
        count = 0
        count1 = 1
        
        for i in d1.columns:
            count1  = 1 + count1
            if count1 != (len(d1.columns)-1):
                if i == 'CDL3BLACKCROWS':
                    count +=  1
                
                    st.write(i ,': ''Overall performance is outstanding. '
                          
                          'Candles in a downward price trend will qualify. The pattern acts as a bearish reversal of the upward price trend and the overall performance is outstanding.'
                
                'A check of the performance over 10 days shows some startling results: Do not trade this if the breakout is downward. Only upward breakouts are worth considering. If you remove the 10 day restriction, then the worst performance comes from three black crows in a bull market, regardless of the breakout direction. \n')
                if i == 'CDL3INSIDE':
                    count +=  1
                
                    st.write(i ,': ''Rare and Highly reliable'  \
                          'Rhe pattern acts as a bearish reversal 65% of the time. The reason for the comparatively high reversal rate is because price closes near the bottom of the candlestick pattern and all a reversal has to do it post a close below the bottom of the candle pattern. That is much easier to do than close above the other end (the top). \n')
                        
                if i == 'CDL3OUTSIDE':
                    count +=  1
                
                    st.write(i ,': '' Frequent bullish reversal but very works well'  \
                          'The three outside up candlestick acts as a bullish reversal both in theory and in reality. And it does so quite well. It has a high frequency number, so you should be able to find is as often as feathers on a duck. The overall performance is also quite good and that means the price trend, post breakout, is worthwhile if not downright tasty. However, you will want to avoid this candlestick pattern if you hold it for a short term (10 days). Upward breakouts under those conditions do particularly lousy. For longer term holds, avoid those in a bull market after a downward breakout. \n')
                        
                if i == 'CDL3WHITESOLDIERS':
                    count +=  1
                
                    st.write(i ,': '' Bullish reversal 82% of the time,'  \
                          ' It acts as a bullish reversal 82% of the time, ranking 3 out of 103 candlestick types, where 1 is best. or a downward breakout to occur, price would have to make a serious drop and plummet even more to push up the performance score. In fact, the high performance is due to just those factors: few candles with downward breakouts.  \n') 
                if i == 'CDLENGULFING':
                    count +=  1
                
                    st.write(i ,': ''Common'  \
                          'A closer look at the numbers shows that downward breakouts are where this pattern outperforms. The best move 10 days after an upward breakout is a drop of 1.18%. Usually you would see a rise 10 days after an upward breakout but not in this candlestick. Thus, if you are going to rely on this candlestick then look for a downward breakout. The best move appears in a bear market, so that is the way to trade this one. \n') 
                
                if i == 'Respectable':
                    count +=  1
                
                    st.write(i ,': ''The evening doji star is one of the better performing candlestick patterns. It has a high reversal rate, ranking 12th, and the performance over time is respectable, too, but not outstanding. A check of the numbers shows that downward breakouts are weakest, and upward ones are strongest with the bear market/up breakout configuration  doing the best of the bunch.  \n'  \
                          '') 
                
                if i == 'CDLEVENINGSTAR':
                    count +=  1
                
                    st.write(i ,': ''Highly reliable as market reversal'  \
                          'The evening star acts as a bearish reversal of the upward price trend, and when the breakout occurs, hold on for dear life. The overall performance rank is very high, but most of the performance is due to upward breakouts, not downward ones. In fact, you will want to avoid trading the evening star after a downward breakout in a bull market. \n') 
                
                if i == 'CDLHAMMER':
                    count +=  1
                
                    st.write(i ,': ''Highly respected'  \
                          'If you project the height of the candle in the direction of the breakout (candle top for upward breakouts and candle bottom for downward ones), price meets the target 88% of the time, which is very good. The best average move occurs after a downward breakout in a bear market. Price drops an average of 4.12% after a hammer, placing the rank at 48 where 1 is best. That, of course, is just mid range out of the 103 candle types studied. A good performance would be a move of 6% or more. \n')
                        
                
                if i == 'CDLHOMINGPIGEON':
                    count +=  1
                
                    st.write(i ,': ''Realiable'  \
                          'The homing pigeon is supposed to act as a bullish reversal, '
                        '  but testing found that it performs as a bearish continuation 56% of the time. That is what I call "near random" because you won''t'' be able to tell the breakout direction with any certainty. Despite the poor reversal rate, the overall performance is quite good. \n') 
                
                if i == 'CDLINVERTEDHAMMER':
                    count +=  1
                
                    st.write(i ,': ''Reliable for bear continuation'  \
                          'A hammer is a single candle line in a downtrend, but an inverted hammer is a two line candle, also in a downtrend. The inverted hammer is supposed to be a bullish reversal candlestick, but it really acts as a bearish continuation 65% of the time. The overall performance ranks it 6 out of 103 candles, meaning the trend after the candle often results in a good sized move. \n') 
                
                if i == 'CDLLONGLINE':
                    count +=  1
                
                    st.write(i ,': ''Reliable'  \
                          'Above the stomach is a simple candlestick pattern and yet it works very well, functioning as a bullish reversal 66% of the time.  \n') 
                
                if i == 'CDLMORNINGDOJISTAR':
                    count +=  1
                
                    st.write(i ,': ''Reliable'  \
                          'Taking a microscope to the performance numbers shows that downward breakouts outperform upward ones both in the 10-day performance rank and by the average move 10 days after the breakout. If you remove the 10-day restriction and just measure the move to the trend end, you will find that downward breakouts in a bull market are the weak performers, but in a bear market, they are the strongest performers. \n') 
                
                if i == 'CDLMORNINGSTAR':
                    count +=  1
                
                    st.write(i ,': ''Not frequent but highly reliable'  \
                          'If you arbitrarily sell 10 days after the breakout, you will find that the morning star after an upward breakout is the weakest performer. However, just letting the trend end when it ends instead of imposing a time limit shows that upward breakouts have better post-breakout performance than downward ones. That tells me the trend after the breakout from a morning star takes a while to get going but it tends to keep moving up. Patience is probably a good word for what you need when trading this candle pattern. \n') 
                
                if i == 'CDLPIERCING':
                    count +=  1
                
                    st.write(i ,'Higher priority but not as frequent'  
                          'Overall performance is good, too, suggesting the price trend after the breakout is a lasting and profitable one. The piercing pattern does best in a bear market, especially after a downward breakout. Upward breakouts in a bull market are the weakest of the four combinations of bull/bear market and up/down breakout direction. \n') 
                
                if i == 'CDLRISEFALL3METHODS':
                    count +=  1
                
                    st.write(i ,': ''Highly reliable bearish continuation but rare' \
                          'The falling three methods pattern acts as a bearish continuation 71% of the time, giving it a rank of 7 which is very high (1 is best). As I mentioned, it is a rare bird, having a frequency rank of 91 out of 103 candles where 1 means it occurs most often.  \n') 
                
                if i == 'CDLSEPARATINGLINES':
                    count +=  1
                
                    st.write(i ,': ''Reliable for bullish continuation'  \
                          'he bullish separating lines candle pattern acts in theory as it does in reality, that of a bullish continuation 72% of the time. That ranks 6th of out 103 candles, which is very strong. The frequency rank is 76, so this candle pattern will not be easy to spot because it is rare.  \n') 
                
                if i == 'CDLSPINNINGTOP':
                    count +=  1
                
                    st.write(i ,': ''Most frequence'  \
                          'As I mentioned in the introduction, the black spinning top works in theory as it does in reality: indecision. Price breaks out in any direction almost equally. With a frequency rank of 1, the candle is prolific, so you will see it often in a historical price series. The best move 10 days after a breakout occurs when price drops 3.36% in a bear market.  \n') 
        
        
    except:
            
            st.markdown("For the nature of the data it is not possible to show the ChandlestickChart")

    
             
if nav == "Statistics Indicators":
    
    st.header('Statistics Indicators')
    
    st.markdown("""
    ** Commodity channel index  
    
    **Fibunachi Retracement
    
    **Ease of Movement
    
    **Moving average
    
    ** Simple Moving Average
    
    **Bollinger Bands
    """)

    TICK = Selected_stock
    data  =  f.baja(TICK)
    dataframes  =   f.dataframes1(data) 
    option = st.radio( "Please choose the data yo want to predict :",('Close', 'Open', 'High','Low'))
    print(option)
    #def_seasonal(option, dataframes,TICK,freq_des)
    st.write("We will calculate th stadistic indicators with", option, 'history prices of', TICK)
          
    try:
        ndays = st.slider('Choose the number of days:', 0, 50, 10)
        
        st.header('Commodity channel index')
        TP = (dataframes['High'] + dataframes['Low'] + dataframes['Close']) / 3 
        CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
            name = 'CCI') 
        dataframes['CCI'] = CCI  
        CCI = dataframes['CCI'].dropna()
        
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
                
        plt.savefig('./imag/CC1.png')
        image16 = Image.open('./imag/CC1.png')
        st.image(image16, caption='')
        
        st.header('Fibunachi Retracement')
        
        
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
        
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(suby_close, color='black')
        ax.axhspan(level1, price_min, alpha=0.4, color='lightsalmon')
        ax.axhspan(level2, level1, alpha=0.5, color='palegoldenrod')
        ax.axhspan(level3, level2, alpha=0.5, color='palegreen')
        ax.axhspan(price_max, level3, alpha=0.5, color='powderblue')
        plt.title('Retracement levels for Fibonacci ratios of 23.6%, 38.2% and 61.8%   ' + TICK)
        plt.ylabel("Price")
        plt.xlabel("Time")
        plt.legend("close")
        plt.show()
        plt.savefig('./imag/fibu.png')
        image17 = Image.open('./imag/fibu.png')
        st.image(image17, caption='')
        
        st.header('Moving average')
        n = st.slider( "Please choose the days for the simple moving average :",20,60,20)
        ew =  st.slider( "Please choose the days for the exponentially Weighted Moving Averag :",50,300,200 )
        #ndays = 50  
        SMA_NIFTY = SMA(dataframes,n)
        SMA_NIFTY = SMA_NIFTY.dropna()
        SMA = SMA_NIFTY['SMA']
        
        EWMA_NIFTY = EWMA(dataframes,ew)
        EWMA_NIFTY = EWMA_NIFTY.dropna()
        EWMA = EWMA_NIFTY['EWMA_200']
        
        # Plotting the NIFTY Price Series chart and Moving Averages below
        plt.figure(figsize=(7,5))
        plt.plot(dataframes['Close'],lw=1, label='NSE Prices')
        plt.plot(SMA,'g',lw=1, label=f'{n}-day SMA (green)')
        plt.plot(EWMA,'r', lw=1, label=f'{ew}-day EWMA (red)')
        plt.legend(loc=2,prop={'size':11})
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True)
        plt.setp(plt.gca().get_xticklabels(), rotation=30)
        
        plt.show()
        plt.savefig('./imag/ma.png')
        image18 = Image.open('./imag/ma.png')
        st.image(image18, caption='')
           
        
        st.header('Ease of Movement')
        n1 = st.slider( "Please choose the days for the ease of movement :",20,60,20)
    
        EVM = EVM(dataframes, n1)
        EVM = EVM['EVM']
        
        # Plotting the Price Series chart and the Ease Of Movement below
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(2, 1, 1)
        ax.set_xticklabels([])
        plt.plot(dataframes['Close'],lw=1)
        plt.title(' Price Chart')
        plt.ylabel('Price')
        plt.grid(True)
        bx = fig.add_subplot(2, 1, 2)
        plt.plot(EVM,'k',lw=0.75,linestyle='-',label=f'EVM({n1})')
        plt.legend(loc=2,prop={'size':9})
        plt.ylabel('EVM values')
        plt.grid(True)
        plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.show()
        plt.savefig('./imag/em.png')
        image19 = Image.open('./imag/em.png')
        st.image(image19, caption='')
        
        st.header('Bollinger Bands')
        nb = st.slider( "Please choose the days for the moving average (Bollinger Bands) :",50,100,200)
          
        NIFTY_BBANDS = BBANDS(dataframes, nb,option)
        pd.concat([NIFTY_BBANDS[option],NIFTY_BBANDS.UpperBB,NIFTY_BBANDS.LowerBB],axis=1).plot(figsize=(9,5),grid=True)
        plt.title(' Price Chart')
        plt.ylabel('Price')
        
        plt.savefig('./imag/bb.png')
        image20 = Image.open('./imag/bb.png')
        st.image(image20, caption='')
        
        
        
       
        

        
     
    except:     
        st.markdown("For the nature of the data it is not possible to calculate the Stadistic Indicators")
         