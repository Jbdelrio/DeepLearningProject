import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from lstmmodel import LSTMModel
from bilstmmodel import BiLSTMModel
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Téléchargement des données de Yahoo Finance
ticker = 'IBM'
data = yf.download(ticker, start='2009-07-01', end='2019-07-01')
data = data[['Adj Close']]  # On conserve uniquement la colonne "Adjusted Close"

# Préparation des données en les normalisant entre 0 et 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Division des données en ensembles d'entraînement et de test (70% train, 30% test)
train_size = int(len(scaled_data) * 0.7)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

# Fonction pour créer les séquences de données
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10  # Nombre de pas de temps à considérer pour prédire le suivant
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Conversion des données en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Fonction d'entraînement des modèles
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
            print(f'Epoch {epoch} Loss: {single_loss.item()}')

# Instanciation et entraînement des modèles LSTM et BiLSTM
lstm_model = LSTMModel()
bilstm_model = BiLSTMModel()


print("Training LSTM Model")
train_model(lstm_model, X_train, y_train)

print("Training BiLSTM Model")
train_model(bilstm_model, X_train, y_train)

# Évaluation des modèles
def evaluate_model(model, test_data):
    model.eval()  # Mode évaluation
    predictions = []
    with torch.no_grad():
        for i in range(len(test_data)):
            seq = test_data[i:i+1]
            y_pred = model(seq).squeeze()  # Enlever la dimension supplémentaire
            predictions.append(y_pred.item())
    return predictions

# Prédictions avec les modèles LSTM et BiLSTM
lstm_predictions = evaluate_model(lstm_model, X_test)
bilstm_predictions = evaluate_model(bilstm_model, X_test)

# Inverser la normalisation pour obtenir les valeurs réelles
lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
bilstm_predictions = scaler.inverse_transform(np.array(bilstm_predictions).reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))

# Calcul du RMSE pour chaque modèle
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
bilstm_rmse = np.sqrt(mean_squared_error(y_test, bilstm_predictions))

print(f'LSTM RMSE: {lstm_rmse}')
print(f'BiLSTM RMSE: {bilstm_rmse}')

# Visualisation des résultats
plt.figure(figsize=(14, 5))
plt.plot(data.index[train_size+look_back+1:], y_test, label='Actual')
plt.plot(data.index[train_size+look_back+1:], lstm_predictions, label='LSTM Predictions')
plt.plot(data.index[train_size+look_back+1:], bilstm_predictions, label='BiLSTM Predictions')
plt.legend()
plt.show()
