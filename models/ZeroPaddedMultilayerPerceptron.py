from torch import nn

from models.abstract.ZeroPaddedRegressor import ZeroPaddedRegressor


class ZeroPaddedMultilayerPerceptron(ZeroPaddedRegressor):
    def __init__(self, hidden_size: int = 64, **_):
        from Config import Config
        super().__init__()
        self.hidden_size = hidden_size
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.max_n_task = Config.n_tasks
        self.dense1 = nn.Linear(
            in_features=(self.max_n_task + 1) * Config.m_machines, out_features=hidden_size
        )
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=1)

    def predict(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return self.relu(x)

    def __str__(self):
        return type(self).__name__ + str(self.hidden_size)
