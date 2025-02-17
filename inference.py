import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from model import LSTM, Transformer  # Ou Transformer si besoin
from test import StockPriceInference  
from pretreatment import StockDataProcessor
import torch

class MainWindow:
    def __init__(self):
        print("Initialisation de l'interface principale...")  # Debug
        self.window = tk.Tk()
        self.window.title('Stock Price Prediction')
        self.window.geometry('800x700')

        self.model_path = tk.StringVar(value='models/best_model_LSTM.h5')
        self.days_to_predict = tk.IntVar(value=7)  # Par défaut, 7 jours
        # Charger le modèle
        print("Chargement du modèle LSTM...")  # Debug
        self.model = LSTM(input_dim=1, hidden_dim=256, num_layers=4, seq_lenght=50)  # Exemple pour LSTM
        
        print("Initialisation de la classe StockPriceInference...")  # Debug
        self.inference = StockPriceInference(
            model=self.model,
            model_path=self.model_path.get(),
            scaler_path='scaler/scaler.pkl',
            data_folder='datas/validation_data',
            column_name='Open',
            seq_length=50,
        )

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

        self.predict_button = ttk.Button(self.window, text="Predict", command=self.predict_stock)
        self.predict_button.pack(pady=20)

        # Zone pour afficher le graphique
        self.canvas_frame = tk.Frame(self.window)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

    def predict_stock(self):
        print("Bouton Predict cliqué.")  # Debug
        ticker = self.ticker_entry.get()
        print(f"Ticker entré : {ticker}")  # Debug
        print(self.days_to_predict)

        # Mise à jour des chemins et des paramètres
        self.inference.model_path = self.model_entry.get()
        print(self.inference.model_path)
        
        if not ticker:
            self.show_error("Please enter a valid ticker.")
            return

        try:
            print(f"Téléchargement des données pour le ticker {ticker}...")  # Debug
            df = yf.download(ticker, period='1y', interval='1d')
            print(f"Données téléchargées : {len(df)} lignes.")  # Debug

            if df.empty:
                self.show_error("No data found for this ticker.")
                return

            # Sauvegarder les données pour passer par le pipeline existant
            data_path = "datas/inference_data/temp.csv"
            df.to_csv(data_path)
            print(f"Données sauvegardées dans {data_path}.")  # Debug

            # Charger et modifier le fichier CSV
            df = pd.read_csv(data_path)
            df = df.drop([0], axis=0)  
            df = df.drop([1], axis=0)  
            df.to_csv(data_path, index=False)

            # Réinitialiser le Data Processor
            print("Réinitialisation de StockDataProcessor...")  # Debug
            self.inference.data_processor = StockDataProcessor(
                data_folder="datas/inference_data",
                column_name="Open",
                seq_length=50,
                is_train=False
            )

            # Réinitialiser les poids du modèle
            print("Réinitialisation du modèle...")  # Debug
            self.inference.model.load_state_dict(torch.load(self.model_entry.get(), map_location=self.inference.device))
            self.inference.model.eval()

            # Appeler la méthode plot_predictions
            print("Appel de la méthode plot_predictions...")  # Debug
            fig = self.inference.plot_predictions(steps=self.days_to_predict.get())
            if not fig:
                self.show_error("Failed to generate the plot.")
                return

            # Nettoyer l'ancien canvas
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            # Intégrer le graphique matplotlib dans Tkinter
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

#models/lstm/lstm_model_20250217_010334.h5
