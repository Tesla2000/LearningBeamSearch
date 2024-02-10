from torch import nn

from regression_models.abstract.BaseRegressor import BaseRegressor


class MultilayerPerceptron(BaseRegressor):
    learning_rate = 1e-5

    def __init__(self, n_tasks: int, m_machines: int, hidden_size: int = 256, n_hidden=0, **_):
        super(MultilayerPerceptron, self).__init__()
        self.hidden_size = hidden_size
        self.flatten = nn.Flatten()
        self.first_dense = nn.Linear(
            in_features=(n_tasks + 1) * m_machines, out_features=hidden_size
        )
        self.hidden_layers = tuple(nn.Linear(in_features=hidden_size, out_features=hidden_size) for _ in range(n_hidden))
        self.last_dense = nn.Linear(in_features=hidden_size, out_features=1)

    def predict(self, x):
        x = self.flatten(x)
        x = self.first_dense(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_dense(x)

    def __str__(self):
        return type(self).__name__ + str(self.hidden_size)
