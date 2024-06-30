import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lstmmodel import LSTMModel
from bilstmmodel import BiLSTMModel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


"""
Lien vers le papier de recherche : https://par.nsf.gov/servlets/purl/10186554
Dans ce script, nous reproduisons le papier de Siami-Namini, Tavakoli et Siami Namin. 
"""


# todo: Téléchargement des données de Yahoo Finance où on ne garde que la colonne "Adjusted Close"
def download_data(ticker, start, end, interval):
    data = yf.download(ticker, start=start, end=end, interval=interval)
    data = data[['Adj Close']]
    return data


# todo: Préparation des données en les normalisant entre 0 et 1
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1,1))
    return scaled_data, scaler


# todo: Fonction pour créer les séquences de données de taille égale à look_back
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


ticker = 'IBM'
start = '2009-07-01'
end = '2019-07-01'
time_interval = '1d' #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
data = download_data(ticker, start, end, time_interval)
scaled_data, scaler = preprocess_data(data)


look_back = 10
X, y = create_dataset(data, look_back)

# todo: Division des données en ensembles d'entraînement et de test (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
print('x_train.shape = ',X_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',X_test.shape)
print('y_test.shape = ',y_test.shape)

# todo: Conversion des données en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# todo: Fonction d'entraînement des modèles
def train_model(model, train_data, train_labels, num_epochs=100, lr=0.001):
    loss_function = nn.MSELoss()  # Fonction de perte : Mean Squared Error
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Optimiseur : Adam
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(train_data).squeeze()  # Enlever la dimension supplémentaire
        single_loss = loss_function(y_pred, train_labels)
        single_loss.backward()
        optimizer.step()
        if epoch % 25 == 0:
            print(f'Epoch {epoch}/{num_epochs} Loss: {single_loss.item()}')


# todo: Instanciation et entraînement des modèles LSTM et BiLSTM
lstm_model = LSTMModel()
# todo : L'innovation proposée avec le BiLSTM par rapport au LSTM est l'entraînement du modèle
#  qui va de l'input à l'output puis repasse de l'output à l'input.
bilstm_model = BiLSTMModel()


print("Training LSTM Model")
train_model(lstm_model, X_train, y_train)

print("Training BiLSTM Model")
train_model(bilstm_model, X_train, y_train)


# todo: Évaluation des modèles
def evaluate_model(model, test_data):
    model.eval()  # Mode évaluation
    predictions = []
    with torch.no_grad():
        for i in range(len(test_data)):
            seq = test_data[i:i+1]
            y_pred = model(seq).squeeze()  # Enlever la dimension supplémentaire
            predictions.append(y_pred.item())
        #accuracy = accuracy_score(y_test, y_pred)
        #print(f'Accuracy: {accuracy * 100:.2f}%')
    return predictions


# todo: Prédictions avec les modèles LSTM et BiLSTM
lstm_predictions = evaluate_model(lstm_model, X_test)
bilstm_predictions = evaluate_model(bilstm_model, X_test)

# todo: Inverser la normalisation pour obtenir les valeurs réelles
lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
bilstm_predictions = scaler.inverse_transform(np.array(bilstm_predictions).reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

# todo: Calcul du RMSE pour chaque modèle
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
bilstm_rmse = np.sqrt(mean_squared_error(y_test, bilstm_predictions))

print(f'LSTM RMSE: {lstm_rmse}')
print(f'BiLSTM RMSE: {bilstm_rmse}')
print(f'RMSEs reduction bring by the BiLSTM: {((bilstm_rmse-lstm_rmse)/lstm_rmse) * 100:.2f}%')


# todo: Visualisation des résultats
train_size = len(X_train)
test_size = len(X_test)

plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size+look_back+1:train_size+look_back+1+test_size], y_test, label='Actual')
plt.plot(data.index[train_size+look_back+1:train_size+look_back+1+test_size], lstm_predictions, label='LSTM Predictions')
plt.plot(data.index[train_size+look_back+1:train_size+look_back+1+test_size], bilstm_predictions, label='BiLSTM Predictions')
plt.legend()
plt.show()