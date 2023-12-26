from torch import nn


class DenseModel(nn.Module):
    def __init__(self, rows: int, n_machines: int):
        super(DenseModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout()
        self.dense1 = nn.Linear(in_features=rows * n_machines, out_features=224)
        self.drop2 = nn.Dropout(.25)
        self.dense2 = nn.Linear(in_features=224, out_features=1)

    def forward(self, x):
        x = x.float()
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        x = self.flatten(x)
        x = self.drop1(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.dense2(x)
        return x

