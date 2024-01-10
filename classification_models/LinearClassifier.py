from abc import abstractmethod

import numpy as np
import torch
from torch import nn, Tensor

from classification_models.BaseClassifier import BaseClassifier
from regression_models.abstract.BaseRegressor import BaseRegressor


class LinearClassifier(BaseClassifier):
    def __init__(
        self, model_regressor: BaseRegressor, learning_rate: float = 1e-4, **_
    ):
        super(LinearClassifier, self).__init__(model_regressor, learning_rate)
        self.fc = nn.Linear(1, 1)

    def _predict(self, x, bound):
        difference = torch.sub(bound.reshape(-1, 1), x)
        difference = torch.div(difference, 100)
        return self.sigmoid(self.fc(difference))
