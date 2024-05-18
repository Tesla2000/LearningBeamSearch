from models import ConvRegressor
from models.abstract.ZeroPaddedRegressor import ZeroPaddedRegressor


class ZeroPaddedConvRegressor(ZeroPaddedRegressor, ConvRegressor):
    def predict(self, x):
        return ConvRegressor.predict(self, x)


if __name__ == '__main__':
    import os

    import torch
    from torchviz import make_dot

    model_type = ZeroPaddedConvRegressor
    model = model_type(50, 10)
    x = torch.randn(1, 51, 10)
    y = model(x)
    make_dot(y, params=dict(list(model.named_parameters()))).render(model_type.__name__, format="png")
    os.system(f"dot -Tpng {model_type.__name__} -o network_images/{model_type.__name__}.png")
    os.remove(model_type.__name__)
    os.remove(model_type.__name__ + ".png")
