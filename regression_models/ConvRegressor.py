from torch import nn
import torch.nn.functional as F
from regression_models.abstract.BaseRegressor import BaseRegressor


class ConvRegressor(BaseRegressor):
    def __init__(self, *args):
        super(ConvRegressor, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=9, kernel_size=3, padding="same"
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=9, out_features=1)

    def predict(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        return self.fc(x)

    def __str__(self):
        return type(self).__name__
