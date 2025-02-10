import torch
import os
import matplotlib.pyplot as plt
from pretreatment import StockDataProcessor
from model import Transformer, LSTM
import joblib

class StockPriceInference:
    def __init__(self, model, model_path='models/best_model.h5', scaler_path='models/scaler.pkl', data_folder='datas/validation_data', column_name='Open', seq_length=50, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)  # Load the scaler
        self.data_processor = StockDataProcessor(data_folder, column_name=column_name, seq_length=seq_length, is_train=False)

    def predict(self):
        self.data_processor.load_data()
        predictions = {}
        for filename, data in zip(os.listdir(self.data_processor.data_folder), self.data_processor.datas):
            sequences = self.data_processor.create_sequences(data)
            sequences = torch.tensor(sequences, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                preds = self.model(sequences).cpu().numpy().flatten()
            preds = self.scaler.inverse_transform(preds.reshape(-1, 1)).flatten()  # Rescale predictions
            predictions[filename] = (data[-len(preds):], preds)  # Store actual data and predictions
        return predictions

    def plot_predictions(self):
        predictions = self.predict()
        plt.figure(figsize=(12, 6))
        
        for filename, (actual, preds) in predictions.items():
            plt.plot(actual, color='blue', label='Actual')
            plt.plot(preds, color='red', linestyle='dashed', label='Predicted')
            plt.legend()
            plt.title(f'Predictions for {filename}')
            plt.show()

# Exemple d'utilisation :
model = LSTM()  # Ou Transformer()
inference = StockPriceInference(model)
inference.plot_predictions()
