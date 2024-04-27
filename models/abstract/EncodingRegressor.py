from abc import abstractmethod, ABC


from models.EncodingNetwork import EncodingNetwork
from models.abstract.BaseRegressor import BaseRegressor


class EncodingRegressor(BaseRegressor, ABC):
    def __init__(self):
        from Config import Config
        super().__init__()
        self.encoder = EncodingNetwork(Config.n_tasks, Config.m_machines).to(Config.device)

    def forward(self, x):
        x = x.float()
        x = self.encoder(x)
        x = self.predict(x)
        return x

    @abstractmethod
    def predict(self, x):
        pass
