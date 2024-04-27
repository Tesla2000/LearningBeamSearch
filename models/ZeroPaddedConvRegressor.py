from models import ConvRegressor
from models.abstract.ZeroPaddedRegressor import ZeroPaddedRegressor


class ZeroPaddedConvRegressor(ZeroPaddedRegressor, ConvRegressor):
    def predict(self, x):
        return ConvRegressor.predict(self, x)
