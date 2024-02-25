from torch import nn

from regression_models.EncodingNetwork import EncodingNetwork, encoder
from regression_models.abstract.EncodingRegressor import EncodingRegressor


class EncodingPerceptron(EncodingRegressor):
    def __init__(self, **_):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(encoder.fc_out_features * encoder.out_channels, 1)
        self.relu = nn.ReLU()

    def predict(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return self.relu(x)
