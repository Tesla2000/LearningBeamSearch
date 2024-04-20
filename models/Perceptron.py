from torch import nn

from models.abstract.BaseRegressor import BaseRegressor


class Perceptron(BaseRegressor):
    def __init__(self, n_tasks: int, m_machines: int, **_):
        super(Perceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear((n_tasks + 1) * m_machines, 1)
        self.relu = nn.LeakyReLU()

    def predict(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return self.relu(x)


class GenPerceptron(Perceptron):
    pass


if __name__ == '__main__':
    import os

    import torch
    from torchviz import make_dot

    model = Perceptron(50, 10)
    x = torch.randn(1, 51, 10)
    y = model(x)
    make_dot(y, params=dict(list(model.named_parameters()))).render(Perceptron.__name__, format="png")
    os.system(f"dot -Tpng {Perceptron.__name__} -o network_images/{Perceptron.__name__}.png")
    os.remove(Perceptron.__name__)
    os.remove(Perceptron.__name__ + ".png")
