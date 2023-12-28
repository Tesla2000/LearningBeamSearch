import torch
from torch import nn


class GRUModel(nn.Module):
    def __init__(self, n_machines: int, num_layers: int = 2, hidden_size: int = 256, **_):
        super(GRUModel, self).__init__()
        self.n_machines = n_machines
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(n_machines, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.float()
        min_value = x[0, 0, 0]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        x, _ = self.gru(x, h0)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x + min_value

