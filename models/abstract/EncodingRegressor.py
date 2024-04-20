from abc import abstractmethod, ABC

from models.EncodingNetwork import encoder
from models.abstract.BaseRegressor import BaseRegressor


class EncodingRegressor(BaseRegressor, ABC):
    def __init__(self):
        super().__init__()
        self._encoder = encoder

    def forward(self, x):
        x = x.float()
        x = self._encoder(x)
        x = self.predict(x)
        return x

    @abstractmethod
    def predict(self, x):
        pass
