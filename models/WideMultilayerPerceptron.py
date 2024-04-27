import torch
from torch import nn

from models import MultilayerPerceptron


class WideMultilayerPerceptron(MultilayerPerceptron):
    def __init__(self, n_tasks: int, m_machines: int, hidden_size: int = 64, **_):
        super(WideMultilayerPerceptron, self).__init__(n_tasks, m_machines, hidden_size)
        self.dense2 = nn.Linear(
            in_features=hidden_size + (n_tasks + 1) * m_machines, out_features=1
        )

    def predict(self, x):
        x = self.flatten(x)
        out = self.dense1(x)
        x = torch.concat((x, out), dim=1)
        x = self.relu(x)
        x = self.dense2(x)
        return self.relu(x)


if __name__ == '__main__':
    import os

    import torch
    from torchviz import make_dot

    model = WideMultilayerPerceptron(50, 10)
    x = torch.randn(1, 51, 10)
    y = model(x)
    make_dot(y, params=dict(list(model.named_parameters()))).render(WideMultilayerPerceptron.__name__, format="png")
    os.system(
        f"dot -Tpng {WideMultilayerPerceptron.__name__} -o network_images/{WideMultilayerPerceptron.__name__}.png")
    os.remove(WideMultilayerPerceptron.__name__)
    os.remove(WideMultilayerPerceptron.__name__ + ".png")
