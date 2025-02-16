import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim:int=1, model_dim:int=64, num_heads:int=4, num_layers:int=3, ff_dim:int=128, dropout:int=0.1, seq_length:int=50):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.seq_length = seq_length
        
        # Embedding layer to project input to model_dim
        self.embedding = nn.Linear(input_dim, model_dim)  
        
        # Positional encoding (define max sequence length as 50 for your data)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, model_dim))  # Assuming max sequence length = 50
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer to predict a single value (price)
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, x):
        # Ensure the input shape is (batch_size, seq_length, input_dim)
        # Apply embedding and positional encoding
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        
        # Rearrange to (seq_length, batch_size, model_dim) for the transformer
        x = x.permute(1, 0, 2)  # Transformer expects (seq_length, batch_size, model_dim)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Use the output from the last time step (x[-1])
        x = x[-1, :, :]  # Shape (batch_size, model_dim)
        
        # Final fully connected layer to predict stock price
        x = self.fc_out(x)  # Output a single value per batch
        
        return x

class LSTM(nn.Module):
    def __init__(self, input_dim:int=1, hidden_dim:int=128, num_layers:int=4, dropout:int=0.1, seq_lenght:int=100):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_lenght = seq_lenght

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True, dropout=dropout)

        self.fc_out = nn.Linear(hidden_dim,1)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out,_ = self.lstm(x,(h0,c0))

        out = out[:, -1, :]

        out = self.fc_out(out)

        return out