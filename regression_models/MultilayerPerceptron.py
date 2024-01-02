from torch import nn

from regression_models.abstract.BaseRegressor import BaseRegressor


class MultilayerPerceptron(BaseRegressor):
    def __init__(self, n_tasks: int, n_machines: int, hidden_size: int = 256, **_):
        super(MultilayerPerceptron, self).__init__()
        self.hidden_size = hidden_size
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(
            in_features=(n_tasks + 1) * n_machines, out_features=hidden_size
        )
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=1)

    def predict(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

    def __str__(self):
        return type(self).__name__ + str(self.hidden_size)
