import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from statsmodels.tsa.arima.model import ARIMA

# Charger les données
df = yf.download('IBM', start='2009-07-01', end='2019-07-31')
data = df['Close'].values.reshape(-1, 1)

# Normaliser les données
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Diviser les données en ensembles d'entraînement et de test (70/30)
training_data_len = int(np.ceil(len(data) * 0.70))
train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len:]

# Fonction pour créer un dataset avec des timestamps look_back
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Reshaper les entrées pour être [échantillons, pas de temps, caractéristiques]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Fonction pour ajuster le modèle LSTM ou BiLSTM
def fit_lstm_bilstm(train, epochs, neurons, option):
    model = Sequential()
    if option == 'L':
        model.add(LSTM(neurons, return_sequences=True, input_shape=(look_back, 1)))
    elif option == 'B':
        model.add(Bidirectional(LSTM(neurons, return_sequences=True, input_shape=(look_back, 1))))
    model.add(LSTM(neurons, stateful=False))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(epochs):
        model.fit(train[0], train[1], epochs=1, shuffle=False)
    return model

# Fonction pour faire des prédictions avec LSTM/BiLSTM
def forecast_lstm(model, test_data):
    predictions = []
    for i in range(len(test_data)):
        X = test_data[i]
        X = np.reshape(X, (1, X.shape[0], 1))
        yhat = model.predict(X, batch_size=1)
        predictions.append(yhat[0][0])
    return predictions

# Fonction pour calculer le RMSE
def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

# Ajuster le modèle LSTM
lstm_model = fit_lstm_bilstm((X_train, y_train), epochs=1, neurons=4, option='L')
lstm_predictions = forecast_lstm(lstm_model, X_test)
lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
lstm_rmse = calculate_rmse(scaler.inverse_transform(test_data[look_back+1:]), lstm_predictions)

# Ajuster le modèle BiLSTM
bilstm_model = fit_lstm_bilstm((X_train, y_train), epochs=1, neurons=4, option='B')
bilstm_predictions = forecast_lstm(bilstm_model, X_test)
bilstm_predictions = scaler.inverse_transform(np.array(bilstm_predictions).reshape(-1, 1))
bilstm_rmse = calculate_rmse(scaler.inverse_transform(test_data[look_back+1:]), bilstm_predictions)

# Ajuster le modèle ARIMA
def fit_arima(train, test):
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    return predictions

arima_predictions = fit_arima(train_data.flatten(), test_data.flatten())
arima_predictions = np.array(arima_predictions).reshape(-1, 1)
arima_predictions = scaler.inverse_transform(arima_predictions)
arima_rmse = calculate_rmse(scaler.inverse_transform(test_data), arima_predictions)

# Calculer la réduction en % du RMSE
def calculate_reduction(new_value, original_value):
    return (new_value - original_value) / original_value * 100

reduction_lstm_arima = calculate_reduction(lstm_rmse, arima_rmse)
reduction_bilstm_arima = calculate_reduction(bilstm_rmse, arima_rmse)
reduction_bilstm_lstm = calculate_reduction(bilstm_rmse, lstm_rmse)

# Imprimer le RMSE et les réductions
print(f"LSTM RMSE: {lstm_rmse}")
print(f"BiLSTM RMSE: {bilstm_rmse}")
print(f"ARIMA RMSE: {arima_rmse}")
print(f"Reduction LSTM over ARIMA: {reduction_lstm_arima:.2f}%")
print(f"Reduction BiLSTM over ARIMA: {reduction_bilstm_arima:.2f}%")
print(f"Reduction BiLSTM over LSTM: {reduction_bilstm_lstm:.2f}%")

# Vérifier quelques prédictions
print("Sample LSTM Predictions:", lstm_predictions[:10])
print("Sample BiLSTM Predictions:", bilstm_predictions[:10])
print("Sample ARIMA Predictions:", arima_predictions[:10])

# Convertir les indices en tableaux numpy
train_index = df.index[:training_data_len].to_numpy()
test_index = df.index[training_data_len:].to_numpy()

# Afficher les résultats
plt.figure(figsize=(14,5))
plt.plot(train_index, scaler.inverse_transform(train_data), label='Train Data')
plt.plot(test_index, scaler.inverse_transform(test_data), label='Test Data')
plt.plot(test_index[look_back+1:], lstm_predictions, label='LSTM')
plt.plot(test_index[look_back+1:], bilstm_predictions, label='BiLSTM')
plt.plot(test_index, arima_predictions, label='ARIMA')
plt.legend()
plt.show()
