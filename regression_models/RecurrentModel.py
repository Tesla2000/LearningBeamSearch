from itertools import combinations

from torch import nn, Tensor
from torch.nn.modules.module import T

from regression_models.EncodingNetwork import EncodingNetwork
from regression_models.abstract.BaseRegressor import BaseRegressor


class RecurrentModel(BaseRegressor):
    def __init__(self, encoder: EncodingNetwork):
        super().__init__()
        self.encoder = encoder
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=self.encoder.m_machines,
                          hidden_size=self.encoder.fc_out_features * self.encoder.out_channels)
        self.fc = nn.Linear(in_features=self.encoder.fc_out_features * self.encoder.out_channels, out_features=1)
        self._states = {}
        self._working_times = {}
        self._working_indexes = {}

    def fill_state(self, working_time_matrix):
        self._states = {tuple(): self.encoder(Tensor(working_time_matrix).unsqueeze(0)).flatten().unsqueeze(0)}
        self._working_times = {}
        self._working_indexes = {}
        for index, row in enumerate(working_time_matrix):
            self._working_indexes[tuple(row)] = index
            self._working_times[index] = Tensor(row).unsqueeze(0)

    def forward(self, x, hx=None):
        preset_indexes = set(map(self._working_indexes.get, map(tuple, x.numpy()[0, 1:])))
        absent_indexes = set(range(len(self._working_indexes))) - preset_indexes
        for index in absent_indexes:
            if (indexes := tuple(sorted(absent_indexes.copy() - {index}))) in self._states:
                prev_state = self._states[indexes]
                row = self._working_times[index]
                absent_indexes = tuple(sorted(absent_indexes))
                _, self._states[absent_indexes] = self.gru(row, prev_state)
                x = self.fc(self._states[absent_indexes])
                x = self.relu(x)
                return x
        for n_elements in range(len(absent_indexes) - 1, -1, -1):
            for combination in combinations(absent_indexes, n_elements):
                if combination in self._states:
                    absent_indexes -= set(combination)
                    prev_state = self._states[combination]
                    for index in absent_indexes:
                        row = self._working_times[index]
                        combination = tuple(sorted(set(combination).union({index})))
                        _, self._states[combination] = self.gru(row, prev_state)
                        prev_state = self._states[combination]
                    x = self.fc(prev_state)
                    x = self.relu(x)
                    return x

    def eval(self: T) -> T:
        super().eval()
        self.encoder.eval()
        return self

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.encoder.train()
        return self
