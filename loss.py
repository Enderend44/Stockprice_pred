import torch
import torch.nn as nn

class LossFunctions:
    @staticmethod
    def mse_loss():
        """Retourne la fonction de perte MSE (Mean Squared Error)."""
        return nn.MSELoss()

    @staticmethod
    def mae_loss():
        """Retourne la fonction de perte MAE (Mean Absolute Error)."""
        return nn.L1Loss()

    @staticmethod
    def huber_loss(delta=1.0):
        """Retourne la fonction de perte Huber."""
        return nn.HuberLoss(delta=delta)

    @staticmethod
    def custom_loss():
        """Retourne une fonction de perte personnalisée (exemple)."""
        # Implémentation d'une perte personnalisée (exemple)
        def loss_fn(output, target):
            return torch.mean(torch.abs(output - target)) + torch.mean((output - target)**2)
        return loss_fn
