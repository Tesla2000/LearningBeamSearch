from abc import abstractmethod

import numpy as np
import torch
from torch import nn, Tensor

from classification_models.BaseClassifier import BaseClassifier
from regression_models.abstract.BaseRegressor import BaseRegressor


class LinearClassifier(BaseClassifier):

    def __init__(self, model_regressor: BaseRegressor, n_tasks: int, learning_rate: float = 1e-4, **_):
        super(LinearClassifier, self).__init__(model_regressor, n_tasks, learning_rate)
        self.fc = nn.Linear(n_tasks, n_tasks)
        self.softmax = nn.Softmax()

    def _predict(self, x):
        x = self.fc(x)
        return self.softmax(x)
