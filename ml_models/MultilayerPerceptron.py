from torch import nn

from ml_models.abstract.DropoutModel import DropoutModel


class MultilayerPerceptron(DropoutModel):
    def __init__(self, n_tasks: int, n_machines: int, **_):
        super(MultilayerPerceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=(n_tasks + 1) * n_machines, out_features=224)
        self.dense2 = nn.Linear(in_features=224, out_features=1)

    def predict(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
