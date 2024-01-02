import numpy as np
import torch
from torch import nn, Tensor

from regression_models.abstract.BaseRegressor import BaseRegressor


class BaseClassifier(nn.Module):

    def __init__(self, model_regressor: BaseRegressor, n_tasks: int, learning_rate: float = 1e-4, **_):
        super(BaseClassifier, self).__init__()
        self.model_regressor = model_regressor
        self.model_regressor.eval()
        self.learning_rate = learning_rate
        self.n_tasks = n_tasks
        self.fc = nn.Linear(n_tasks, n_tasks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        outputs = []
        for sample in x:
            regression_predictions = []
            for child_index in range(1, len(sample)):
                x = np.array(sample)
                child = sample[child_index]
                header = np.zeros_like(child)
                header[0] = child[0]
                for i in range(1, len(child)):
                    header[i] = max(header[i - 1], sample[0, i]) + child[i]
                header -= header[0]
                x[0] = header
                left_to_choose = list(range(len(x)))
                left_to_choose.pop(child_index)
                x = Tensor(x[left_to_choose]).unsqueeze(0)
                regression_predictions.append(float(self.model_regressor(x)))
            outputs.append(self.sigmoid(self.fc(Tensor(regression_predictions))))
        return torch.stack(outputs)





    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return self.__str__()
