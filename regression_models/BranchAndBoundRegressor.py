import torch
from torch import nn

from regression_models.abstract.BaseRegressor import BaseRegressor


class BranchAndBoundRegressor(BaseRegressor):
    def __init__(self, **_):
        super(BranchAndBoundRegressor, self).__init__()

    def predict(self, x):
        return 0
