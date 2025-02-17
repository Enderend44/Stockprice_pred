import torch
import os
import matplotlib.pyplot as plt
from pretreatment import StockDataProcessor
from model import Transformer, LSTM
import joblib
import numpy as np


class StockPriceInference:
    def __init__(self, model, model_path='models/best_model.h5', scaler_path='scaler/scaler.pkl', data_folder='datas/validation_data', column_name='Open', seq_length=50, device=None):
        # Configuration du modèle et du dispositif
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Charger le scaler utilisé pour normaliser les données d'entraînement
        self.scaler = joblib.load(scaler_path)

        # Prétraitement des données
        self.data_processor = StockDataProcessor(data_folder, column_name=column_name, seq_length=seq_length, is_train=False)

    def denormalize(self, normalized_data):
        """ Dénormalisation en inversant le MinMaxScaler et en appliquant log inverse. """
        denormalized_data = self.scaler.inverse_transform(normalized_data.reshape(-1, 1))
        return np.expm1(denormalized_data).flatten()

    def predict(self):
        """ Effectuer les prédictions pour les données de validation. """
        self.data_processor.load_data()
        self.data_processor.normalize_all_data()
        predictions = {}

        # Générer les données pour chaque fichier dans le dossier
        for filename, data in zip(os.listdir(self.data_processor.data_folder), self.data_processor.datas):
            # Générer les séquences et les cibles
            sequences, actual_values, last_sequence = self.data_processor.create_sequences(data)

            # Conversion en tenseur
            sequences = torch.tensor(sequences, dtype=torch.float32).to(self.device)

            # Prédictions
            with torch.no_grad():
                preds = self.model(sequences).cpu().numpy().flatten()

            # Dénormaliser les prédictions et les valeurs réelles
            preds = self.denormalize(preds)
            actual_values = self.denormalize(actual_values)

            # Stocker les données réelles, prédites et la dernière séquence pour chaque fichier
            predictions[filename] = {
                "actual_values": actual_values,
                "predictions": preds,
                "last_sequence": last_sequence
            }

        return predictions

    def iterative_prediction(self, last_sequence, steps=7):
        """Faire des prédictions itératives à partir de la dernière séquence connue."""
        preds = []
        current_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(self.device)

        for _ in range(steps):
            # Ajouter une dimension batch
            current_sequence = current_sequence.unsqueeze(0)

            # Prédire la prochaine valeur
            with torch.no_grad():
                next_value = self.model(current_sequence).cpu().numpy().flatten()

            # Stocker la prédiction
            preds.append(next_value[0])

            # Mettre à jour la séquence courante avec la nouvelle prédiction
            next_value_tensor = torch.tensor(next_value, dtype=torch.float32).unsqueeze(-1).to(self.device)
            current_sequence = torch.cat((current_sequence[0, 1:], next_value_tensor), dim=0).detach()

        # Dénormaliser les prédictions
        preds = self.denormalize(np.array(preds))
        return preds

    def plot_predictions(self, steps=7):
        """Générer des graphiques pour comparer les prédictions avec les valeurs réelles."""
        predictions = self.predict()

        plt.figure(figsize=(12, 6))

        for filename, data in predictions.items():
            actual_values = data["actual_values"]
            preds = data["predictions"]
            last_sequence = data["last_sequence"]

            # Prédictions itératives
            future_preds = self.iterative_prediction(last_sequence, steps=steps)

            # Tracer les valeurs réelles et prédites
            plt.plot(actual_values, color='blue', label='Actual')
            plt.plot(preds, color='red', linestyle='dashed', label='Predicted')

            # Tracer les prédictions itératives
            plt.plot(range(len(actual_values), len(actual_values) + len(future_preds)), future_preds, color='green', linestyle='dotted', label='Iterative Predictions')

            plt.legend()
            plt.title(f'Predictions for {filename}')
            plt.xlabel("Time Steps")
            plt.ylabel("Stock Price")
            #plt.show()

        return plt.gcf() , actual_values, preds


# Exemple d'utilisation
if __name__ == "__main__":
    # model = Transformer(seq_length=200)
    model = LSTM(input_dim=1, hidden_dim=256, num_layers=4, seq_lenght=200)  # Transformer(input_dim=1, seq_length=200) # Peut aussi être un Transformer
    inference = StockPriceInference(
        model,
        model_path='models/best_model_LSTM.h5',
        scaler_path='scaler/scaler.pkl',
        data_folder='datas/validation_data'
    )
    fig,_,_ = inference.plot_predictions()
    plt.show()
