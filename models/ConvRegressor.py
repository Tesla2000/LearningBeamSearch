from torch import nn
import torch.nn.functional as F

from models.abstract.BaseRegressor import BaseRegressor


class ConvRegressor(BaseRegressor):
    def __init__(self, *args):
        super(ConvRegressor, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=3, padding="same"
        )
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=9, kernel_size=3, padding="same"
        )
        self.leaky_relu = nn.LeakyReLU()
        self.fc = nn.Linear(in_features=9, out_features=1)

    def predict(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(-1)
        x = x.squeeze(-1)
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
