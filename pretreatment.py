#%%
import numpy as np
import pandas as pd 
import random as rd
import os 
import torch
from sklearn.preprocessing import MinMaxScaler
import pickle

class StockDataProcessor:
    def __init__(self, data_folder: str, column_name: str = 'Open', batch_size: int = 1000, seq_length: int = 50, is_train=True, scaler_path="models/scaler.pkl"):
        self.data_folder = data_folder
        self.column_name = column_name
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.datas = []
        self.is_train = is_train
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if not self.is_train and os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                #print("Scaler chargé depuis", self.scaler_path)

    def load_data(self):
        for path, dirs, files in os.walk(self.data_folder):
            for filename in files:
                df = pd.read_csv(os.path.join(path, filename))
                df = df[self.column_name].dropna()  # Supprime les valeurs NaN
                if self.is_train:
                    df = self.scaler.fit_transform(df.values.reshape(-1, 1)).flatten()
                    with open(self.scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    #print(f"Scaler sauvegardé dans {self.scaler_path}")
                else:
                    df = self.scaler.transform(df.values.reshape(-1, 1)).flatten()

                self.datas.append(df)

        return self.datas

    def create_sequences(self, dataset):
        """ Create sequences for Transformer model input. """
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
        #print(f"Sequences created: Shape = {sequences.shape}, Targets Shape = {targets.shape}")
        return sequences, targets

    def data_generator(self):
        while True:
            if not self.datas:
                raise ValueError("No data loaded. Please run load_data() first.")

            dataset = rd.choice(self.datas)

            if len(dataset) < self.seq_length + 1:
                continue
            
            sequences, targets = self.create_sequences(dataset)

            start_idx = rd.randint(0, len(sequences) - self.batch_size)
            batch_x = sequences[start_idx:start_idx + self.batch_size]
            batch_y = targets[start_idx:start_idx + self.batch_size]

            #print(f"Generated batch: X Shape = {batch_x.shape}, Y Shape = {batch_y.shape}, Batch Size = {len(batch_x)}")  # Vérification

            yield torch.tensor(batch_x, dtype=torch.float32), torch.tensor(batch_y, dtype=torch.float32)  # Format PyTorch

"""
#%%
# Test pour StockDataProcessor
if __name__ == "__main__":
    data_folder = "datas/training_data"  # Dossier contenant les données
    processor = StockDataProcessor(data_folder, column_name='Open', batch_size=1000, seq_length=50)
    
    print("Loading training data...")
    processor.load_data()

    print("\nGenerating sequences and batches...")
    generator = processor.data_generator()
    
    # Test d'un batch
    batch_x, batch_y = next(generator)
    
    print(f"Batch X: Shape = {batch_x.shape}, Batch Size = {len(batch_x)}")
    print(f"Batch Y: Shape = {batch_y.shape}, Batch Size = {len(batch_y)}")

# %%
"""