from abc import abstractmethod, ABC

from torch import nn

from regression_models.abstract.BaseRegressor import BaseRegressor


class BaseClassifier(nn.Module, ABC):

    def __init__(self, model_regressor: BaseRegressor, n_tasks: int, learning_rate: float = 1e-4, **_):
        super(BaseClassifier, self).__init__()
        self.model_regressor = model_regressor
        self.model_regressor.eval()
        self.learning_rate = learning_rate
        self.n_tasks = n_tasks

    # def _get_outputs(self, x):
    #     outputs = []
    #     for sample in x:
    #         regression_predictions = []
    #         for child_index in range(1, len(sample)):
    #             x = np.array(sample)
    #             child = sample[child_index]
    #             header = np.zeros_like(child)
    #             header[0] = child[0]
    #             for i in range(1, len(child)):
    #                 header[i] = max(header[i - 1], sample[0, i]) + child[i]
    #             header -= header[0]
    #             x[0] = header
    #             left_to_choose = list(range(len(x)))
    #             left_to_choose.pop(child_index)
    #             x = Tensor(x[left_to_choose]).unsqueeze(0)
    #         #     regression_predictions.append(float(self.model_regressor(x)))
    #         # outputs.append(Tensor(regression_predictions))
    #             regression_predictions.append(self.model_regressor(x)[0, 0])
    #         outputs.append(torch.stack(regression_predictions))
    #     return torch.stack(outputs)

    @abstractmethod
    def _predict(self, x, bound):
        pass

    def forward(self, x, bound):
        x = self.model_regressor(x)
        return self._predict(x, bound).reshape(-1, 1)

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return self.__str__()
