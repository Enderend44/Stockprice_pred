import torch
import os
import matplotlib.pyplot as plt
from pretreatment import StockDataProcessor
from model import Transformer, LSTM
import joblib

class StockPriceInference:
    def __init__(self, model, model_path='models/best_model.h5', scaler_path='models/scaler.pkl', data_folder='datas/validation_data', column_name='Open', seq_length=50, device=None):
        # Configuration du modèle et du dispositif
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Charger le scaler utilisé pour normaliser les données d'entraînement
        self.scaler = joblib.load(scaler_path)

        # Prétraitement des données
        self.data_processor = StockDataProcessor(data_folder, column_name=column_name, seq_length=seq_length, is_train=False)

    def predict(self):
        """ Effectuer les prédictions pour les données de validation. """
        self.data_processor.load_data()
        predictions = {}

        # Générer les données pour chaque fichier dans le dossier
        for filename, data in zip(os.listdir(self.data_processor.data_folder), self.data_processor.datas):
            # Générer les séquences et les cibles
            sequences, actual_values = self.data_processor.create_sequences(data)  # Inclut x (features) et y (targets)
            
            # Conversion en tenseur
            sequences = torch.tensor(sequences, dtype=torch.float32).to(self.device)

            # Prédictions
            with torch.no_grad():
                preds = self.model(sequences).cpu().numpy().flatten()

            # Ramener les prédictions et les valeurs réelles à leur échelle d'origine
            preds = self.scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
            actual_values = self.scaler.inverse_transform(actual_values.reshape(-1, 1)).flatten()

            # Stocker les données réelles et prédites pour chaque fichier
            predictions[filename] = (actual_values, preds)
        return predictions

    def plot_predictions(self):
        """ Générer des graphiques pour comparer les prédictions avec les valeurs réelles. """
        predictions = self.predict()
        plt.figure(figsize=(12, 6))
        
        for filename, (actual, preds) in predictions.items():
            plt.plot(actual, color='blue', label='Actual')  # Valeurs réelles
            plt.plot(preds, color='red', linestyle='dashed', label='Predicted')  # Valeurs prédites
            plt.legend()
            plt.title(f'Predictions for {filename}')
            plt.xlabel("Time Steps")
            plt.ylabel("Stock Price")
            plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    model = Transformer(input_dim=1, seq_length=200) #LSTM(input_dim=1, hidden_dim=256, num_layers=5, seq_lenght=200)  # Peut aussi être un Transformer
    inference = StockPriceInference(
        model, 
        model_path='models/best_model.h5', 
        scaler_path='models/scaler.pkl', 
        data_folder='datas/validation_data'
    )
    inference.plot_predictions()
