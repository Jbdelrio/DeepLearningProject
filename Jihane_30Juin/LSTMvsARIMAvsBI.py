

import warnings
warnings.filterwarnings('ignore')

# Data wrangling & pre-processing
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
import yfinance as yf

### Downloading IBM Stock Data


# Fetch IBM stock data from Yahoo Finance
df = yf.download('IBM', start='2009-07-01', end='2019-07-31')


### Exploratory Data Analysis (EDA)


# Plotting the Open prices
df['Open'].plot(label='IBM', figsize=(15, 7))
plt.title('Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plotting the Volume of stocks traded
df['Volume'].plot(label='IBM', figsize=(15, 7))
plt.title('Volume of Stock traded')
plt.xlabel('Time')
plt.ylabel('Volume')
plt.legend()
plt.show()

# Market Capitalization
df['MarketCap'] = df['Open'] * df['Volume']
df['MarketCap'].plot(label='IBM', figsize=(15, 7))
plt.title('Market Cap')
plt.xlabel('Time')
plt.ylabel('Capitalization')
plt.legend()
plt.show()

# Stock Volatility
df['returns'] = (df['Close'] / df['Close'].shift(1)) - 1
df['returns'].hist(bins=100, label='IBM', alpha=0.5, figsize=(15, 7))
plt.title('Stock Volatility')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.show()


### LSTM Model for Stock Price Prediction

# Prepare data for LSTM model
data1 = df[['Close']].dropna()
trainData = data1.iloc[:, 0:1].values

# Scaling the data
sc = MinMaxScaler(feature_range=(0, 1))
trainData = sc.fit_transform(trainData)

# Creating training data
X_train, y_train = [], []
for i in range(60, len(trainData)):
    X_train.append(trainData[i-60:i, 0])
    y_train.append(trainData[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# LSTM Model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
hist = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2)

# Preparing test data
testData = df[['Close']].dropna()
y_test = testData.iloc[60:, 0:1].values
inputClosing = testData.iloc[:, 0:1].values
inputClosing_scaled = sc.transform(inputClosing)

X_test = []
for i in range(60, len(testData)):
    X_test.append(inputClosing_scaled[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predicting stock prices
y_pred = model.predict(X_test)
predicted_price = sc.inverse_transform(y_pred)

# Plotting the results
plt.plot(y_test, color='black', label='Actual Stock Price')
plt.plot(predicted_price, color='red', label='Predicted Stock Price')
plt.title('IBM Stock Price Prediction - LSTM Model')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plotting the loss
plt.plot(hist.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# Calculating RMSE
MSE_error = mean_squared_error(y_test, predicted_price)
print('Testing Mean Squared Error LSTM is {}'.format(MSE_error))
print('The Root Mean Squared Error LSTM is:')
lstm_rmse=math.sqrt(MSE_error)
print(lstm_rmse)
### BI-LSTM Model for Stock Price Prediction


# BI-LSTM Model
model_bi = Sequential()
model_bi.add(Bidirectional(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1))))
model_bi.add(Dropout(0.2))
model_bi.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model_bi.add(Dropout(0.2))
model_bi.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model_bi.add(Dropout(0.2))
model_bi.add(Bidirectional(LSTM(units=64, return_sequences=False)))
model_bi.add(Dropout(0.2))
model_bi.add(Dense(units=1))

model_bi.compile(optimizer='adam', loss='mean_squared_error')
hist_bi = model_bi.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2)

# Predicting stock prices using BI-LSTM
y_pred_bi = model_bi.predict(X_test)
predicted_price_bi = sc.inverse_transform(y_pred_bi)

# Plotting the results
plt.plot(y_test, color='black', label='Actual Stock Price')
plt.plot(predicted_price_bi, color='red', label='Predicted Stock Price')
plt.title('IBM Stock Price Prediction - BI-LSTM Model')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

# Plotting the loss
plt.plot(hist_bi.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# Calculating RMSE
MSE_error_bi = mean_squared_error(y_test, predicted_price_bi)
print('Testing Mean Squared Error BiLSTM is {}'.format(MSE_error_bi))
print('The Root Mean Squared Error BiLSTM is:')
bilstm_rmse=math.sqrt(MSE_error_bi)
print(bilstm_rmse)


### ARIMA Model for Stock Price Prediction


from statsmodels.tsa.arima.model import ARIMA

# Splitting the data
train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]
training_data = train_data['Close'].values
test_data = test_data['Close'].values

history = [x for x in training_data]
model_predictions = []
N_test_observations = len(test_data)

# ARIMA Model
for time_point in range(N_test_observations):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = test_data[time_point]
    history.append(true_test_value)

# Calculating RMSE
MSE_error_arima = mean_squared_error(test_data, model_predictions)
print('Testing Mean Squared Error ARIMA is {}'.format(MSE_error_arima))
print('The Root Mean Squared Error ARIMA is:')
arima_rmse=math.sqrt(MSE_error_arima)
print(arima_rmse)


# Calculer la réduction en % du RMSE
def calculate_reduction(new_value, original_value):
    return (new_value - original_value) / original_value * 100

reduction_lstm_arima = calculate_reduction(lstm_rmse, arima_rmse)
reduction_bilstm_arima = calculate_reduction(bilstm_rmse, arima_rmse)
reduction_bilstm_lstm = calculate_reduction(bilstm_rmse, lstm_rmse)

print(f"Reduction LSTM over ARIMA: {reduction_lstm_arima:.2f}%")
print(f"Reduction BiLSTM over ARIMA: {reduction_bilstm_arima:.2f}%")
print(f"Reduction BiLSTM over LSTM: {reduction_bilstm_lstm:.2f}%")

# Plotting the results
test_set_range = df[int(len(df)*0.7):].index.to_numpy()
plt.plot(test_set_range, model_predictions, color='blue', marker='o', label='Predicted Price')
plt.plot(test_set_range, test_data, color='red', label='Actual Price')
plt.title('IBM Stock Price')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.show()