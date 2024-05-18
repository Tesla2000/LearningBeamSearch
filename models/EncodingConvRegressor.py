from models import ConvRegressor
from models.abstract.EncodingRegressor import EncodingRegressor


class EncodingConvRegressor(EncodingRegressor, ConvRegressor):

    def forward(self, x):
        return EncodingRegressor.forward(self, x)

    def predict(self, x):
        return ConvRegressor.predict(self, x)


if __name__ == '__main__':
    import os

    import torch
    from torchviz import make_dot

    model_type = EncodingConvRegressor
    model = model_type()
    x = torch.randn(1, 2, 10)
    y = model(x)
    make_dot(y, params=dict(list(model.named_parameters()))).render(model_type.__name__, format="png")
    os.system(f"dot -Tpng {model_type.__name__} -o network_images/{model_type.__name__}.png")
    os.remove(model_type.__name__)
    os.remove(model_type.__name__ + ".png")
