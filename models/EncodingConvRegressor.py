from models import ConvRegressor
from models.abstract.EncodingRegressor import EncodingRegressor


class EncodingConvRegressor(EncodingRegressor, ConvRegressor):

    def forward(self, x):
        return EncodingRegressor.forward(self, x)

    def predict(self, x):
        return ConvRegressor.predict(self, x)
