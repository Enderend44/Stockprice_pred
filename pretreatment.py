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
        self.is_train = is_train  # Indique si on est en mode entraînement ou test
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if not self.is_train and os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                print("Scaler chargé depuis", self.scaler_path)

    def load_data(self):
        for path, dirs, files in os.walk(self.data_folder):
            for filename in files:
                df = pd.read_csv(os.path.join(path, filename))
                df = df[self.column_name].dropna()  # Supprime les valeurs NaN
                print('is_train:', self.is_train)

                if self.is_train:
                    df = self.scaler.fit_transform(df.values.reshape(-1, 1)).flatten()
                    with open(self.scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    print(f"Scaler sauvegardé dans {self.scaler_path}")
                    print(f"Data loaded (training) for {filename}: Shape = {df.shape}, Length = {len(df)}")

                else:
                    df = self.scaler.transform(df.values.reshape(-1, 1)).flatten()
                    print(f"Data loaded (test) for {filename}: Shape = {df.shape}, Length = {len(df)}")

                self.datas.append(df)

        return self.datas


    
    def create_sequences(self, dataset):
        """ Create sequences for Transformer model input. """
        sequences = []
        for i in range(len(dataset) - self.seq_length):
            seq = dataset[i:i + self.seq_length]
            sequences.append(seq)

        sequences = np.array(sequences)
        sequences = np.expand_dims(sequences, axis=-1)  # Ajoute une dimension pour feature_dim

        print(f"Sequences created: Shape = {sequences.shape}, Length = {len(sequences)}")
        return sequences


    def data_generator(self):
        while True:
            if not self.datas:
                raise ValueError("No data loaded. Please run load_data() first.")

            dataset = rd.choice(self.datas)

            if len(dataset) < self.seq_length:
                continue
            
            sequences = self.create_sequences(dataset)

            start_idx = rd.randint(0, len(sequences) - self.batch_size)
            batch = sequences[start_idx:start_idx + self.batch_size]

            print(f"Generated batch: Shape = {batch.shape}, Batch Size = {len(batch)}")  # Vérification

            yield torch.tensor(batch, dtype=torch.float32)  # Format PyTorch


"""
#%%
# Test for StockDataProcessor on "datas/training_data"
if __name__ == "__main__":
    data_folder = "datas/training_data"  # The folder containing your stock data
    processor = StockDataProcessor(data_folder, column_name='Open', batch_size=1000, seq_length=50)
    
    print("Loading training data...")
    # Load the data (training data)
    processor.load_data(is_train=True)
    
    print("\nGenerating sequences and batches...")
    # Generate a batch of data
    generator = processor.data_generator()
    
    # Fetch a single batch to test
    batch = next(generator)
    
    # Print the shape and size of the batch
    print(f"Batch generated: Shape = {batch.shape}, Batch Size = {len(batch)}")

# %%
"""