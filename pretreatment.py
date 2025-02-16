import numpy as np
import pandas as pd
import random as rd
import os
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle


class StockDataProcessor:
    def __init__(self, data_folder:str, column_name: str = 'Open', batch_size: int = 1000, seq_length: int = 50, is_train:bool=True, scaler_path:str="scaler/scaler.pkl"):
        self.data_folder = data_folder
        self.column_name = column_name
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.datas = []
        self.is_train = is_train
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Charger le scaler si on est en mode validation/test
        if not self.is_train and os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

    def load_data(self):
        """Charge les données depuis le dossier spécifié."""
        try:
            for path, dirs, files in os.walk(self.data_folder):
                for filename in files:
                    if filename.endswith('.csv'):
                        df = pd.read_csv(os.path.join(path, filename))
                        df = df[self.column_name].dropna()  # Supprime les valeurs NaN

                        # Applique la transformation logarithmique
                        df_log = np.log1p(df.values)  # log(prices + 1)
                        self.datas.append(df_log)
        except Exception as e:
            print(f"Erreur lors du chargement des données : {e}")

        return self.datas

    def normalize_all_data(self):
        """Applique la normalisation à toutes les données chargées."""
        if not self.datas:
            raise ValueError("No data loaded. Please run load_data() first.")

        normalized_datas = []

        if self.is_train:
            # Ajuster le scaler sur les données concaténées d'entraînement
            all_data = np.concatenate(self.datas).reshape(-1, 1)
            self.scaler.fit(all_data)
            
            # Sauvegarder le scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

        # Appliquer la normalisation à chaque série
        for data in self.datas:
            normalized_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
            normalized_datas.append(normalized_data)

        # Remplacer les données brutes par les données normalisées
        self.datas = normalized_datas

    def create_sequences(self, dataset):
        """Crée des séquences pour l'entrée du modèle Transformer."""
        sequences = []
        targets = []
    
        for i in range(len(dataset) - self.seq_length):
            seq = dataset[i:i + self.seq_length]
            target = dataset[i + self.seq_length]  # y = x_t
            sequences.append(seq)
            targets.append(target)
    
        sequences = np.array(sequences)
        targets = np.array(targets)
        sequences = np.expand_dims(sequences, axis=-1)  # Ajoute une dimension pour feature_dim
    
        # Retourne également la dernière séquence pour les prédictions itératives
        last_sequence = dataset[-self.seq_length:]
        last_sequence = np.expand_dims(last_sequence, axis=-1)  # Ajoute aussi feature_dim pour correspondre au modèle
    
        return sequences, targets, last_sequence


    def data_generator(self):
        """Générateur pour produire des batchs de données."""
        while True:
            if not self.datas:
                raise ValueError("No data loaded. Please run load_data() and normalize_all_data() first.")

            dataset = rd.choice(self.datas)

            if len(dataset) < self.seq_length + 1:
                continue

            sequences, targets, _ = self.create_sequences(dataset)

            start_idx = rd.randint(0, len(sequences) - self.batch_size)
            batch_x = sequences[start_idx:start_idx + self.batch_size]
            batch_y = targets[start_idx:start_idx + self.batch_size]

            yield torch.tensor(batch_x, dtype=torch.float32), torch.tensor(batch_y, dtype=torch.float32)
