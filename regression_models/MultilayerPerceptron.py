from torch import nn

from regression_models.abstract.BaseRegressor import BaseRegressor


class MultilayerPerceptron(BaseRegressor):
    learning_rate = 1e-5

    def __init__(self, n_tasks: int, m_machines: int, hidden_size: int = 64, **_):
        super(MultilayerPerceptron, self).__init__()
        self.hidden_size = hidden_size
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.dense1 = nn.Linear(
            in_features=(n_tasks + 1) * m_machines, out_features=hidden_size
        )
        self.dense2 = nn.Linear(in_features=hidden_size, out_features=1)

    def predict(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return self.relu(x)

    def __str__(self):
        return type(self).__name__ + str(self.hidden_size)


if __name__ == '__main__':
    import os

    import torch
    from torchviz import make_dot

    model = MultilayerPerceptron(50, 10)
    x = torch.randn(1, 51, 10)
    y = model(x)
    make_dot(y, params=dict(list(model.named_parameters()))).render(MultilayerPerceptron.__name__, format="png")
    os.system(f"dot -Tpng {MultilayerPerceptron.__name__} -o network_images/{MultilayerPerceptron.__name__}.png")
    os.remove(MultilayerPerceptron.__name__)
    os.remove(MultilayerPerceptron.__name__ + ".png")
