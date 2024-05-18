import torch.nn.functional as F
from torch import nn
import torch

from models.abstract.BaseRegressor import BaseRegressor


class ConvRegressorAnySize(BaseRegressor):
    def __init__(self, *args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=9, kernel_size=3, padding="same"
        )
        self.leaky_relu = nn.LeakyReLU()
        self.fc = nn.Linear(in_features=10, out_features=1)

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
        x = torch.concat((x, torch.full((x.shape[0], 1), n_tasks).to(Config.device)), dim=1)
        return self.fc(x)

    def __str__(self):
        return type(self).__name__


if __name__ == '__main__':
    import os

    import torch
    from torchviz import make_dot

    model_type = ConvRegressorAnySize
    model = model_type(50, 10)
    x = torch.randn(1, 51, 10)
    y = model(x)
    make_dot(y, params=dict(list(model.named_parameters()))).render(model_type.__name__, format="png")
    os.system(f"dot -Tpng {model_type.__name__} -o network_images/{model_type.__name__}.png")
    os.remove(model_type.__name__)
    os.remove(model_type.__name__ + ".png")