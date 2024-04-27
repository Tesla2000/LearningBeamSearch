import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
import torch

from models.abstract.BaseRegressor import BaseRegressor


class ConvRegressorAnySizeOneHot(BaseRegressor):
    def __init__(self, *args, n_tasks: int = None):
        super().__init__()
        if n_tasks is None:
            from Config import Config
            n_tasks = Config.n_tasks + 1
        self.encoder_length = n_tasks
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=9, kernel_size=3, padding="same"
        )
        self.leaky_relu = nn.LeakyReLU()
        self.fc = nn.Linear(in_features=9 + n_tasks, out_features=1)

    def predict(self, x):
        from Config import Config

        n_tasks = x.shape[-2]
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = torch.concat((x, torch.Tensor(np.eye(self.encoder_length)[n_tasks - 1]).expand((x.shape[0], self.encoder_length)).to(Config.device)), dim=1)
        return self.fc(x)

    def __str__(self):
        return type(self).__name__


if __name__ == '__main__':
    import os

    import torch
    from torchviz import make_dot

    model = ConvRegressor()
    x = torch.randn(1, 50, 10)
    y = model(x)
    make_dot(y, params=dict(list(model.named_parameters()))).render(ConvRegressor.__name__, format="png")
    os.system(f"dot -Tpng {ConvRegressor.__name__} -o network_images/{ConvRegressor.__name__}.png")
    os.remove(ConvRegressor.__name__)
    os.remove(ConvRegressor.__name__ + ".png")
