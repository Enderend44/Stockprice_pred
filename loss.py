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
    def rmse_loss():
        """Retourne la fonction de perte RMSE (Root Mean Squared Error)."""
        def loss_fn(output, target):
            mse = nn.MSELoss()
            return torch.sqrt(mse(output, target))
        return loss_fn

    @staticmethod
    def log_cosh_loss():
        """Retourne la fonction de perte Log-Cosh."""
        def loss_fn(output, target):
            return torch.mean(torch.log(torch.cosh(output - target)))
        return loss_fn

    @staticmethod
    def smooth_l1_loss():
        """Retourne la fonction de perte Smooth L1."""
        return nn.SmoothL1Loss()

    @staticmethod
    def custom_loss():
        """Retourne une fonction de perte personnalisée (exemple)."""
        # Implémentation d'une perte personnalisée (exemple)
        def loss_fn(output, target):
            return torch.mean(torch.abs(output - target)) + torch.mean((output - target)**2)
        return loss_fn
