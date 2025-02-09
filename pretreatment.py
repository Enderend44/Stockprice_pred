#%%
import numpy as np
import pandas as pd 
import random as rd
import matplotlib.pyplot as plt
import os 
import torch
from sklearn.preprocessing import MinMaxScaler

class StockDataProcessor:
    def __init__(self, data_folder: str, column_name: str = 'Open', batch_size: int = 1000):
        self.data_folder = data_folder
        self.column_name = column_name
        self.batch_size = batch_size
        self.datas = []
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    # Function to load data from the given folder.
    def load_data(self):
        for path, dirs, files in os.walk(self.data_folder):
            for filename in files:
                df = pd.read_csv(os.path.join(path, filename))
                df = df[self.column_name].dropna().reshape(-1, 1)  # Reverse to keep chronological order
                df = self.scaler.fit_transform(df).flatten()  # Normalize using sklearn MinMaxScaler
                self.datas.append(df)
        return self.datas
    
    def data_generator(self):
        while True:
            if not self.datas:
                raise ValueError("No data loaded. Please run load_data() first.")
            
            # Choose a random dataset
            dataset = rd.choice(self.datas)
            
            # Ensure there's enough data for a batch
            if len(dataset) < self.batch_size:
                continue
            
            # Choose a random start index ensuring batch_size constraint
            start_idx = rd.randint(0, len(dataset) - self.batch_size)
            batch = dataset[start_idx:start_idx + self.batch_size]
            
            yield batch  # Yield the batch as a NumPy array

    def plot_batch(self, batch):
        plt.figure(figsize=(10, 5))
        plt.plot(batch, label='Stock Prices')
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title('Stock Price Batch')
        plt.legend()
        plt.show()
    
    def transform_for_transformer(self, batch):
        """ Prepare batch data for a Transformer model in PyTorch."""
        batch = torch.tensor(batch, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        return batch
#%%
# Example usage
def main():
    training_data = StockDataProcessor('datas/training_data')
    training_data.load_data()
    
    generator = training_data.data_generator()
    for _ in range(5):  # Generate 5 batches as a test
        batch = next(generator)
        print(batch[:10])  # Print first 10 values of each batch to check
        training_data.plot_batch(batch)
        transformed_batch = training_data.transform_for_transformer(batch)
        print(transformed_batch.shape)

main()

# %%
