import torch
import torch.nn as nn
from pretreatment import StockDataProcessor
from model import Transformer, LSTM
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

class StockPriceTrainer:
    def __init__(self, model, train_data_folder, test_data_folder, batch_size=1000, epochs=100, lr=0.001, checkpoint_path='models/best_model.h5', log_dir="logs"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = StockDataProcessor(train_data_folder, batch_size=batch_size, is_train=True)
        self.test_data = StockDataProcessor(test_data_folder, batch_size=batch_size, is_train=False)
        self.batch_size = batch_size
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5, verbose=True)
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        generator = self.test_data.data_generator()
        with torch.no_grad():
            for _ in range(len(self.test_data.datas)):
                batch = next(generator).to(self.device)
                target = batch[:, -1, :].to(self.device)  # We want to predict the last value of each sequence
                output = self.model(batch)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(self.test_data.datas)

    def train(self):
        self.train_data.load_data()  # Charge les données d'entraînement
    
        # Associer le scaler de train à test
        self.test_data.scaler = self.train_data.scaler  

        self.test_data.load_data()
        
        for epoch in range(self.epochs):
            generator = self.train_data.data_generator()
            total_loss = 0
            self.model.train()
            for _ in range(len(self.train_data.datas)):
                batch = next(generator).to(self.device)
                target = batch[:, -1, :].to(self.device) # We want to predict the last value of each sequence
                
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_data.datas)
            test_loss = self.evaluate()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            # Log the losses to TensorBoard
            self.writer.add_scalars("Loss", {"Train": avg_loss, "Test": test_loss}, epoch)
            
            # Scheduler to reduce learning rate if validation loss plateaus
            self.scheduler.step(test_loss)
            
            # Check if the test loss improved to save the model
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"New best model saved with test loss: {test_loss:.4f}")
        
        # Close the writer
        self.writer.close()


def main():
    model = LSTM()#Transformer()
    trainer = StockPriceTrainer(model, 'datas/training_data', 'datas/test_data')
    trainer.train()

main()
