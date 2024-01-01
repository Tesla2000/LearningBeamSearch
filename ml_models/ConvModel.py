from torch import nn

from ml_models.abstract.BaseModel import BaseModel


class ConvModel(BaseModel):

    def __init__(self, n_tasks: int, n_machines: int, hidden_size: int = 256):
        super(ConvModel, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(n_tasks, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=(n_tasks, 3), padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=3)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=(n_tasks + 1) * n_machines * 9, out_features=hidden_size)
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=1)

    def predict(self, x):
        if len(x.shape) != 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

    def __str__(self):
        return type(self).__name__ + str(self.hidden_size)
