from torch import nn

from regression_models.abstract.BaseRegressor import BaseRegressor


class Perceptron(BaseRegressor):
    def __init__(self, n_tasks: int, m_machines: int, **_):
        super(Perceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear((n_tasks + 1) * m_machines, 1)
        self.relu = nn.ReLU()

    def predict(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return self.relu(x)
