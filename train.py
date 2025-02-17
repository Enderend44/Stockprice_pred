import argparse
import os
import torch
from datetime import datetime
from pretreatment import StockDataProcessor
from loss import LossFunctions
from model import Transformer, LSTM
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter


class StockPriceTrainer:
    def __init__(self, model, train_data_folder, test_data_folder, batch_size=1000, seq_length=50, epochs=100, lr=0.001, checkpoint_path='models/best_model.h5', log_dir="logs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = StockDataProcessor(train_data_folder, batch_size=batch_size, seq_length=seq_length, is_train=True)
        self.test_data = StockDataProcessor(test_data_folder, batch_size=batch_size, seq_length=seq_length, is_train=False)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5, verbose=True)
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

    def evaluate(self):
        """ Evaluate the model on test data. """
        self.model.eval()
        total_loss = 0
        generator = self.test_data.data_generator()
        with torch.no_grad():
            for _ in range(len(self.test_data.datas)):
                batch_x, batch_y = next(generator)
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output.squeeze(), batch_y)
                total_loss += loss.item()

        return total_loss / len(self.test_data.datas)

class StockPriceTrainer:
    def __init__(self, model, train_data_folder, test_data_folder, batch_size=1000, seq_length=50, epochs=100, lr=0.001, checkpoint_path='models/best_model.h5', log_dir="logs", loss_function="mse_loss"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = StockDataProcessor(train_data_folder, batch_size=batch_size, seq_length=seq_length, is_train=True)
        self.test_data = StockDataProcessor(test_data_folder, batch_size=batch_size, seq_length=seq_length, is_train=False)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.epochs = epochs
        
        # Choisir la fonction de perte à partir de LossFunctions
        loss_fn_class = LossFunctions()
        self.criterion = getattr(loss_fn_class, loss_function)()  # Dynamique pour choisir la fonction de perte
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5, verbose=True)
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

    def evaluate(self):
        """ Evaluate the model on test data. """
        self.model.eval()
        total_loss = 0
        generator = self.test_data.data_generator()
        with torch.no_grad():
            for _ in range(len(self.test_data.datas)):
                batch_x, batch_y = next(generator)
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                output = self.model(batch_x)
                loss = self.criterion(output.squeeze(), batch_y)
                total_loss += loss.item()

        return total_loss / len(self.test_data.datas)

    def train(self):
        """ Train the model. """
        # Charger et normaliser les données d'entraînement
        print("Chargement et normalisation des données d'entraînement...")
        self.train_data.load_data()
        self.train_data.normalize_all_data()  # Appliquer la normalisation sur les données d'entraînement

        # Charger et normaliser les données de test
        print("Chargement et normalisation des données de test...")
        self.test_data.load_data()
        self.test_data.scaler = self.train_data.scaler  # Utiliser le même scaler que pour l'entraînement
        self.test_data.normalize_all_data()

        for epoch in range(self.epochs):
            generator = self.train_data.data_generator()
            total_loss = 0
            self.model.train()

            for _ in range(len(self.train_data.datas)):
                batch_x, batch_y = next(generator)
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output.squeeze(), batch_y)  # Squeeze pour s'assurer que les dimensions correspondent
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_data.datas)
            test_loss = self.evaluate()

            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")

            # Log losses to TensorBoard
            self.writer.add_scalars("Loss", {"Train": avg_loss, "Test": test_loss}, epoch)

            # Réduction du taux d'apprentissage si la perte de validation stagne
            self.scheduler.step(avg_loss)

            # Sauvegarder le modèle si la perte de test s'améliore
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved with test loss: {test_loss:.4f}")

        # Fermer le writer TensorBoard
        self.writer.close()



def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description="Train a stock price prediction model.")
    parser.add_argument("--model", type=str, required=True, choices=["transformer", "lstm"], help="Choose the model to train: 'transformer' or 'lstm'.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--seq_length", type=int, default=100, help="Sequence length for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--loss_function", type=str, default="mse_loss", choices=["mse_loss", "mae_loss", "huber_loss", "custom_loss"], help="Choose the loss function.")
    args = parser.parse_args()

    # Initialiser le modèle selon l'argument
    if args.model == "transformer":
        model = Transformer(input_dim=1, seq_length=args.seq_length)
        checkpoint_folder = "models/transformer/"
    elif args.model == "lstm":
        model = LSTM(input_dim=1, hidden_dim=256, num_layers=4, seq_lenght=args.seq_length)
        checkpoint_folder = "models/lstm/"

    # Créer le dossier de sauvegarde s'il n'existe pas
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Générer un nom de fichier unique pour le modèle
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_folder, f"{args.model}_model_{timestamp}.h5")

    # Initialiser et démarrer l'entraînement
    trainer = StockPriceTrainer(
        model=model,
        train_data_folder='datas/training_data',
        test_data_folder='datas/test_data',
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=checkpoint_path,
        log_dir=f"logs/{args.model}_{timestamp}",
        loss_function=args.loss_function  # Passer le nom de la fonction de perte
    )
    trainer.train()



if __name__ == "__main__":
    main()
