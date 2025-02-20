import torch
import os
import joblib
import numpy as np
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pretreatment import StockDataProcessor
from model import Transformer, LSTM


class ModelEvaluator:
    def __init__(self, data_folder='datas/validation_data', scaler_path='scaler/scaler.pkl', device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = joblib.load(scaler_path)
        self.data_processor = StockDataProcessor(data_folder, column_name='Open', seq_length=50, is_train=False)
    
    def denormalize(self, normalized_data):
        """ Dénormalisation des données en inversant le MinMaxScaler et en appliquant log inverse. """
        denormalized_data = self.scaler.inverse_transform(normalized_data.reshape(-1, 1))
        return np.expm1(denormalized_data).flatten()

    def load_model(self, model_path, model_type='lstm', seq_length=50):
        """ Charger un modèle à partir du fichier avec son architecture. """
        if model_type == 'lstm':
            model = LSTM(input_dim=1, hidden_dim=256, num_layers=4, seq_lenght=seq_length)
        elif model_type == 'transformer':
            model = Transformer(seq_length=seq_length)
            self.adjust_transformer_positional_encoding(model, model_path)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def adjust_transformer_positional_encoding(self, model, model_path):
        """ Ajuster la taille de positional_encoding pour correspondre au modèle sauvegardé. """
        state_dict = torch.load(model_path, map_location=self.device)
        saved_positional_encoding = state_dict['positional_encoding']

        if saved_positional_encoding.size(1) != model.seq_length:
            new_seq_length = saved_positional_encoding.size(1)  # Taille de séquence du modèle sauvegardé
            model.positional_encoding = torch.nn.Parameter(
                torch.nn.functional.interpolate(
                    model.positional_encoding,
                    size=(new_seq_length, model.positional_encoding.size(-1)),
                    mode='linear',
                    align_corners=False
                )
            )
            model.seq_length = new_seq_length  # Met à jour la taille de séquence du modèle

    def evaluate_model(self, model, model_path):
        """ Calculer les scores RMSE et MAE pour le modèle donné. """
        self.data_processor.load_data()
        self.data_processor.normalize_all_data()

        all_preds = []
        all_actual_values = []

        model.to(self.device)

        for filename, data in zip(os.listdir(self.data_processor.data_folder), self.data_processor.datas):
            sequences, actual_values, _ = self.data_processor.create_sequences(data)
            sequences = torch.tensor(sequences, dtype=torch.float32).to(self.device)

            # Prédictions
            with torch.no_grad():
                preds = model(sequences).cpu().numpy().flatten()

            # Dénormaliser les valeurs
            preds = self.denormalize(preds)
            actual_values = self.denormalize(actual_values)

            all_preds.extend(preds)
            all_actual_values.extend(actual_values)

        # Calcul des scores RMSE et MAE
        rmse = np.sqrt(mean_squared_error(all_actual_values, all_preds))
        mae = mean_absolute_error(all_actual_values, all_preds)

        return rmse, mae

    def extract_hyperparameters_from_filename(self, model_filename):
        """ Extraire les hyperparamètres et la fonction de perte du nom du fichier du modèle. """
        # Adapter le regex pour prendre en compte un éventuel préfixe (comme "models/lstm/")
        pattern = r"(?:.*/)?(\w+)_model_(\d{8}_\d{6})_batch(\d+)_seq(\d+)_epochs(\d+)_lr([\d\.]+)_(\w+)_loss\.h5"
        match = re.search(pattern, model_filename)
    
        if match:
            return {
                'model_type': match.group(1),  # 'lstm' ou 'transformer'
                'date': match.group(2),
                'batch_size': int(match.group(3)),
                'seq_length': int(match.group(4)),
                'epochs': int(match.group(5)),
                'learning_rate': float(match.group(6)),
                'loss_function': match.group(7)
            }
        else:
            raise ValueError("Nom du fichier de modèle ne correspond pas au format attendu.")


    def evaluate_all_models(self, models_folder='models', model_type='lstm'):
        """ Évaluer tous les modèles dans le dossier spécifié et retourner les scores associés. """
        scores = []

        for model_filename in os.listdir(os.path.join(models_folder, model_type)):
            model_path = os.path.join(models_folder, model_type, model_filename)
            print(f"Évaluation du modèle: {model_filename}")

            # Extraire les hyperparamètres pour récupérer seq_length
            hyperparameters = self.extract_hyperparameters_from_filename(model_filename)
            seq_length = hyperparameters['seq_length']

            # Charger et évaluer le modèle
            model = self.load_model(model_path, model_type, seq_length=seq_length)
            rmse, mae = self.evaluate_model(model, model_path)
            
            hyperparameters.update({'rmse': rmse, 'mae': mae})
            scores.append(hyperparameters)

        return scores

    def get_best_model(self, scores, metric='rmse'):
        """ Trouver le modèle avec les meilleures performances basé sur le métrique donné. """
        sorted_scores = sorted(scores, key=lambda x: x[metric])  # Trie par RMSE ou MAE
        return sorted_scores[0]  # Retourne le meilleur modèle


# Exemple d'utilisation
if __name__ == "__main__":
    evaluator = ModelEvaluator()

    # Évaluer les modèles LSTM
    lstm_scores = evaluator.evaluate_all_models(model_type='lstm')
    print("Scores LSTM:", lstm_scores)
    best_lstm_model = evaluator.get_best_model(lstm_scores, metric='rmse')
    print("Meilleur modèle LSTM:", best_lstm_model)

    # Évaluer les modèles Transformer
    transformer_scores = evaluator.evaluate_all_models(model_type='transformer')
    print("Scores Transformer:", transformer_scores)
    best_transformer_model = evaluator.get_best_model(transformer_scores, metric='rmse')
    print("Meilleur modèle Transformer:", best_transformer_model)
