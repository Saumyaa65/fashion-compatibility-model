import torch 
import torch.nn as nn

class OutfitLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_layers=1, bidirectional=True):
        super(OutfitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.out_dim, 1)  # For binary compatibility score (sigmoid later)

    def forward(self, x):
        # x shape = [batch, sequence_len, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim*D]
        final_repr = lstm_out[:, -1, :]  # Use the last output in the sequence
        dropped = self.dropout(final_repr)
        score = self.fc(dropped)  # [batch, 1]
        return score
