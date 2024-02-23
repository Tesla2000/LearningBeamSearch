from torch import nn

from regression_models.EncodingNetwork import EncodingNetwork
from regression_models.abstract.BaseRegressor import BaseRegressor


class RecurrentModel(BaseRegressor):
    def __init__(self, encoder: EncodingNetwork, hidden_size: int = 2048):
        super().__init__()
        self.encoder = encoder
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=self.encoder.m_machines, hidden_size=hidden_size)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)
        self._state = None

    def fill_state(self, working_time_matrix):
        self._state = self.encoder(working_time_matrix)

    def forward(self, x, hx=None):
        x = x.float()
        _, self._state = self.gru(x.flatten(), self._state)
        x = self.fc(self._state)
        x = self.relu(x)
        return x
