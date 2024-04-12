from torch import nn

from Config import Config
from regression_models.abstract.ZeroPaddedRegressor import ZeroPaddedRegressor


class ZeroPaddedPerceptron(ZeroPaddedRegressor):
    def __init__(self, **_):
        super(ZeroPaddedPerceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear((self.max_n_task + 1) * Config.m_machines, 1)
        self.relu = nn.LeakyReLU()

    def predict(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return self.relu(x)
