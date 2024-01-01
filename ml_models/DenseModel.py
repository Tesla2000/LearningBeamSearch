from torch import nn

from ml_models.BaseModel import BaseModel


class DenseModel(BaseModel):
    def __init__(self, n_tasks: int, n_machines: int, **_):
        super(DenseModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout()
        self.dense1 = nn.Linear(in_features=(n_tasks + 1) * n_machines, out_features=224)
        self.drop2 = nn.Dropout(.25)
        self.dense2 = nn.Linear(in_features=224, out_features=1)

    def predict(self, x):
        x = self.flatten(x)
        x = self.drop1(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.drop2(x)
        return self.dense2(x)
