import torch
from torch import nn

from ml_models.BaseModel import BaseModel


class MeanModel(BaseModel):
    def __init__(self, **_):
        super(MeanModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def predict(self, x):
        x = torch.mean(x)
        return self.fc(x)
