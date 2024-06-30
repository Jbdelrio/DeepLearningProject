import torch.nn as nn

# Définition du modèle BiLSTM
class BiLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers = 2, output_size=1):
        #todo : Appel au constructeur de la classe parente nn
        super(BiLSTMModel, self).__init__()
        
        #todo : Taille de la couche cachée
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        #todo : Définition de la couche LSTM bidirectionnelle
        # input_size: taille des caractéristiques d'entrée
        # hidden_layer_size: nombre de neurones dans la couche cachée
        # bidirectional=True: LSTM bidirectionnelle (ici le modèle réalise un lSTM simple et répète l'entrainement à l'envers de la sortie à l'entrée)
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, bidirectional=True)
        
        #todo : Définition de la couche linéaire
        # hidden_layer_size * 2: taille d'entrée multipliée par 2 car bidirectionnelle
        # output_size: taille de la sortie
        self.linear = nn.Linear(hidden_layer_size * 2, output_size)

    def forward(self, input_seq):
        #todo : Transformation de l'entrée pour correspondre à la forme attendue par LSTM
        # input_seq.view(len(input_seq), 1, -1): transforme les dimensions de l'entrée
        #lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        
        #todo : Transformation de la sortie de LSTM pour correspondre à la forme attendue par la couche linéaire
        # lstm_out.view(len(input_seq), -1): aplatit la sortie de LSTM
        #predictions = self.linear(lstm_out.view(len(input_seq), -1))

        #todo : Vérifiez la forme de l'entrée, elle doit être (batch_size, sequence_length, input_size)
        batch_size = input_seq.shape[0]
        sequence_length = input_seq.shape[1]

        #todo : Transformation de l'entrée pour correspondre à la forme attendue par LSTM
        # Si l'entrée est de forme (batch_size, sequence_length)
        if len(input_seq.shape) == 2:
            input_seq = input_seq.view(batch_size, sequence_length, -1)

        #todo : Passer à travers la couche LSTM
        lstm_out, _ = self.lstm(input_seq)

        #todo : Prendre la sortie de la dernière étape temporelle
        lstm_out_last = lstm_out[:, -1, :]

        #todo : Passer à travers la couche linéaire
        predictions = self.linear(lstm_out_last)
        
        #todo : Retourne la dernière prédiction
        return predictions[-1]
