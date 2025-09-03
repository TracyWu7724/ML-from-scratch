import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out


class LSTMfromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weights for input, forget, cell, and output gates
        self.W_i = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.W_f = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.W_c = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        self.W_o = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size))
        
        # Biases for the gates
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        all_outputs = []

        
        for t in range(seq_length):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, h_t), dim=1)
            
            i_t = torch.sigmoid(combined @ self.W_i + self.b_i)
            f_t = torch.sigmoid(combined @ self.W_f + self.b_f)
            g_t = torch.tanh(combined @ self.W_c + self.b_c)
            o_t = torch.sigmoid(combined @ self.W_o + self.b_o)
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            all_outputs.append(h_t.unsqueeze(1))
        
        all_outputs = torch.cat(all_outputs, dim=1)  # (batch_size, seq_length, hidden_size)

        out = self.fc(all_outputs[:, -1, :]) 
        return out





