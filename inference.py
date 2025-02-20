import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from model import LSTM, Transformer  # Ou Transformer si besoin
from test import StockPriceInference  
from pretreatment import StockDataProcessor
from evaluator import ModelEvaluator
from loss import *
import torch



class MainWindow:
    def __init__(self):
        print("Initialisation de l'interface principale...")  # Debug
        self.window = tk.Tk()
        self.window.title('Stock Price Prediction')
        self.window.geometry('800x700')

        # Variables Tkinter
        self.model_path = tk.StringVar(value='models/lstm/lstm_model_20250219_230057_batch1024_seq50_epochs500_lr0.001_mae_loss.h5')
        self.days_to_predict = tk.IntVar(value=7)  # Par défaut, 7 jours

        # Instance de ModelEvaluator pour les fonctions liées au modèle
        self.evaluator = ModelEvaluator(scaler_path='scaler/scaler.pkl', data_folder='datas/validation_data')

        # Initialise les hyperparamètres (seront extraits dynamiquement)
        self.hyperparameters = None
        self.model = None
        self.setup_ui()

    def setup_ui(self):
        print("Configuration de l'interface utilisateur...")  # Debug

        # Champ d'entrée pour le ticker
        self.label = tk.Label(self.window, text="Enter Yahoo Finance ticker:", font=("Arial", 14))
        self.label.pack(pady=10)

        self.ticker_entry = tk.Entry(self.window, font=("Arial", 14), width=30)
        self.ticker_entry.pack(pady=10)

        # Champ pour sélectionner le modèle
        self.model_label = tk.Label(self.window, text="Enter model path:", font=("Arial", 14))
        self.model_label.pack(pady=10)

        self.model_entry = tk.Entry(self.window, textvariable=self.model_path, font=("Arial", 14), width=50)
        self.model_entry.pack(pady=10)

        # Champ pour le nombre de jours
        self.days_label = tk.Label(self.window, text="Enter number of days for prediction:", font=("Arial", 14))
        self.days_label.pack(pady=10)

        self.days_entry = tk.Entry(self.window, textvariable=self.days_to_predict, font=("Arial", 14), width=10)
        self.days_entry.pack(pady=10)

        # Bouton prédiction
        self.predict_button = ttk.Button(self.window, text="Predict", command=self.predict_stock)
        self.predict_button.pack(pady=20)

        # Zone pour afficher le graphique
        self.canvas_frame = tk.Frame(self.window)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Label pour afficher la RMSE
        self.rmse_label = tk.Label(self.window, text="", font=("Arial", 12), fg="blue")
        self.rmse_label.pack(pady=10)

    def predict_stock(self):
        print("Bouton Predict cliqué.")  # Debug
        ticker = self.ticker_entry.get()
        model_path = self.model_entry.get()
        days_to_predict = self.days_to_predict.get()

        if not ticker:
            self.show_error("Please enter a valid ticker.")
            return

        try:
            print(f"Extraction des hyperparamètres depuis le modèle : {model_path}...")  # Debug
            self.hyperparameters = self.evaluator.extract_hyperparameters_from_filename(model_path)

            print("Chargement du modèle...")  # Debug
            self.model = self.evaluator.load_model(
                model_path=model_path,
                model_type=self.hyperparameters['model_type'],
                seq_length=self.hyperparameters['seq_length']
            )

            print(f"Modèle chargé : {self.hyperparameters['model_type']} avec longueur de séquence {self.hyperparameters['seq_length']}")  # Debug

            # Initialisation de StockPriceInference avec les nouveaux paramètres
            inference = StockPriceInference(
                model=self.model,
                model_path=model_path,
                scaler_path='scaler/scaler.pkl',
                data_folder='datas/validation_data',
                column_name='Open',
                seq_length=self.hyperparameters['seq_length']
            )

            # Télécharger les données du ticker
            print(f"Téléchargement des données pour le ticker {ticker}...")  # Debug
            df = yf.download(ticker, start="2024-01-01", end="2024-12-31")
            if df.empty:
                self.show_error("No data found for this ticker.")
                return

            # Sauvegarder les données dans un fichier temporaire
            data_path = "datas/inference_data/temp.csv"
            df.to_csv(data_path)
            print(f"Données sauvegardées dans {data_path}.")  # Debug

            # Charger et modifier le fichier CSV
            df = pd.read_csv(data_path)
            df = df.drop([0], axis=0)  
            df = df.drop([1], axis=0)  
            df.to_csv(data_path, index=False)

            # Réinitialiser StockDataProcessor
            inference.data_processor = StockDataProcessor(
                data_folder="datas/inference_data",
                column_name="Open",
                seq_length=self.hyperparameters['seq_length'],
                is_train=False
            )

            # Générer les prédictions et afficher le graphique
            print("Génération des prédictions...")  # Debug
            fig, actual_values, preds = inference.plot_predictions(steps=days_to_predict)

            rmse = LossFunctions.rmse_loss()
            rmse_score = rmse(torch.tensor(preds),torch.tensor(actual_values))

            print(f"RMSE calculée : {rmse_score}")  # Debug

            # Mettre à jour l'interface
            self.rmse_label.config(text=f"RMSE (Erreur quadratique moyenne) : {rmse_score:.4f}")

            # Nettoyer l'ancien graphique
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            # Intégrer le graphique dans l'interface Tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            print(f"Erreur rencontrée : {str(e)}")  # Debug
            self.show_error(f"An error occurred: {str(e)}")

    def show_error(self, message):
        print(f"Affichage d'une erreur : {message}")  # Debug
        error_window = tk.Toplevel(self.window)
        error_window.title("Error")
        tk.Label(error_window, text=message, font=("Arial", 12), fg="red").pack(pady=20)
        ttk.Button(error_window, text="Close", command=error_window.destroy).pack(pady=10)


if __name__ == "__main__":
    print("Lancement de l'application...")  # Debug
    app = MainWindow()
    app.window.mainloop()
