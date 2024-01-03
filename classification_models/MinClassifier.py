from torch import nn

from classification_models.BaseClassifier import BaseClassifier
from regression_models.abstract.BaseRegressor import BaseRegressor


class MinClassifier(BaseClassifier):

    def __init__(self, model_regressor: BaseRegressor, n_tasks: int, learning_rate: float = 1e-4, **_):
        super(MinClassifier, self).__init__(model_regressor, n_tasks, learning_rate)
        self.softmax = nn.Softmax()

    def _predict(self, x, bound):
        x = -x
        return self.softmax(x)
