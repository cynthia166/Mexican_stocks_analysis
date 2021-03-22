# Mexican stocks historical analysis

The main  objective of this project is to create a financial app, where two models are considered  so the app use would be able to test different parameter and be able 
to choose the best model. 

The app gives you the prediction with Neuronal Networks and Time Series. It also gives you important financial indicators and an explicit explanation of the candlestick pattern of each stock!


The mexican stocks that were considered where does who had economic growth despite the pandemic, which are the following:

*FIBRAPL14.MX-FIBRA Prologis is a leading Real Estate Investment Trust

*GNP.MX -Insurance carrier

*ALPEKA.MX- Chemical manufacturing

*GMEXICOB.MX- Main divisions: Mining, transportation and infrastructure

*PE&OLES.MX - Refined silver, metallic bismuth, sodium sulfate, gold, lead and zinc

*BIMBOA.MX- Food

*GRUMAB.MX-Tortilla  and corn

*SITESB-1.MX- Develops installs maintains and operates communication towers

*CADUA.MX- Real Estate

![image](https://user-images.githubusercontent.com/73049364/111925377-45fa4f80-8a6e-11eb-99a4-d25869abb738.png)



## Structur

This app interacts with the user to predict with Two models Autoregressive Moving Average (ARMA) and Long Short-Term Memory Networks (LSTM).

It gives you indicators which demonstrate the trend of the stock and it also has Candlestick Pattern Recognition.

1.Prediction Time Series


2. Prediction Neural Network


3. Statistics Indicators


4. Candlestick pattern recognition

## Prediction Time Series

The app decomposes  each time series in three elements, statinalitty  trend and error.

The available model used was ARMA which is a combination of  p autoregressive terms and q  moving average. This model has the characteristic of having mean 0 and a
variance of sigma.
They can only be applied to the time series that are stationary, we test out data with the adfuller test and if the data is not stationary then we apply a transformation.
For the validation of the model  the root mean squared error is obtained.

![image](https://user-images.githubusercontent.com/73049364/111925676-71ca0500-8a6f-11eb-98c3-a0351475f577.png)
## Neuronal Network

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior.

A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell.

First, we divide the training and the test set.

After we normalize the training set. The LSTM will have time_step = 60 data entrance (consecutive) and a data output. We let the user choose the layer and the neurons and the method to optimize.

Finally we compare the predictions with our original data. For the validation of the model Humbert Loss, is provided.

![image](https://user-images.githubusercontent.com/73049364/111925859-151b1a00-8a70-11eb-99c0-d72bb3fe7332.png)

## Candlestick pattern recognition
Candlestick charts are a technical tool that packs data for multiple time frames into single price bars. We look for bearish or bullish patterns in order to predict based in our past. We obtain the 20th most reliable or important pattern  from the historical data and a brief explanation is provided.

![image](https://user-images.githubusercontent.com/73049364/111926415-5a404b80-8a72-11eb-8704-a85ddd95fe4c.png)


## Statistics Indicators

** Commodity channel index

**Fibunachi Retracement

**Ease of Movement

**Moving average

** Simple Moving Average

**Bollinger Bands


![image](https://user-images.githubusercontent.com/73049364/111926197-77284f00-8a71-11eb-95d3-431e15cfbc33.png) 







