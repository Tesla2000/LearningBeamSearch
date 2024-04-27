from torch import nn

from models.abstract.EncodingRegressor import EncodingRegressor


class EncodingPerceptron(EncodingRegressor):
    def __init__(self, **_):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.encoder.fc_out_features * self.encoder.out_channels, 1)
        self.relu = nn.LeakyReLU()

    def predict(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return self.relu(x)
