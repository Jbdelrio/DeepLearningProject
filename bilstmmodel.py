import torch.nn as nn

# Définition du modèle BiLSTM
class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        # Appel au constructeur de la classe parente nn.Module
        super(BiLSTMModel, self).__init__()
        
        # Taille de la couche cachée
        self.hidden_layer_size = hidden_layer_size
        
        # Définition de la couche LSTM bidirectionnelle
        # input_size: taille des caractéristiques d'entrée
        # hidden_layer_size: nombre de neurones dans la couche cachée
        # bidirectional=True: LSTM bidirectionnelle
        self.lstm = nn.LSTM(input_size, hidden_layer_size, bidirectional=True)
        
        # Définition de la couche linéaire
        # hidden_layer_size * 2: taille d'entrée multipliée par 2 car bidirectionnelle
        # output_size: taille de la sortie
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, input_seq):
        # Transformation de l'entrée pour correspondre à la forme attendue par LSTM
        # input_seq.view(len(input_seq), 1, -1): transforme les dimensions de l'entrée
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        
        # Transformation de la sortie de LSTM pour correspondre à la forme attendue par la couche linéaire
        # lstm_out.view(len(input_seq), -1): aplatit la sortie de LSTM
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        
        # Retourne la dernière prédiction
        return predictions[-1]
