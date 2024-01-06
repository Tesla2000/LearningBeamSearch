from torch import nn

from classification_models.BaseClassifier import BaseClassifier
from regression_models.abstract.BaseRegressor import BaseRegressor


class MinClassifier(BaseClassifier):

    def __init__(self, model_regressor: BaseRegressor, learning_rate: float = 1e-4, **_):
        super(MinClassifier, self).__init__(model_regressor, learning_rate)

    def _predict(self, x, bound):
        return self.sigmoid(bound.reshape(-1, 1) - x)
