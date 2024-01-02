from torch import nn

from ml_models.abstract.BaseModel import BaseModel


class Perceptron(BaseModel):
    learning_rate = 1e-3

    def __init__(self, n_tasks: int, n_machines: int, **_):
        super(Perceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear((n_tasks + 1) * n_machines, 1)

    def predict(self, x):
        x = self.flatten(x)
        return self.fc(x)
