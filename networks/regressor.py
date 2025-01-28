#import torch
import torch.nn as nn
import torch.optim as optim


class Regressor(nn.Module):
    def __init__(self, input_size = 332*100, hidden_size1 = 256, hidden_size2 = 128, output_size = 1):
        super(Regressor, self).__init__()
        
        self.fullyConnectedLayer = nn.Sequential
        (
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size),
        )

    def forward(self, x):
        return self.fullyConnectedLayer(x)