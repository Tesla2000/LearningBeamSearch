from pathlib import Path
from time import time

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn, Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader

from Config import Config
from regression_models.RegressionDataset import RegressionDataset
from regression_models.abstract.BaseRegressor import BaseRegressor


class SumRegressor(BaseRegressor):
    learning_rate = 1e-3

    def __init__(self, **_):
        super(SumRegressor, self).__init__()
        self.fc = nn.Linear(1, 1)

    def predict(self, x):
        x = torch.sum(x[:, 1:, :-1], dim=[1, 2]).reshape(-1, 1).float()
        return self.fc(x)


if __name__ == '__main__':
    n_machines = 25
    for n_tasks in range(3, 11):
        data_file = Path(f"{Config.TRAINING_DATA_REGRESSION_PATH}/{n_tasks}_{n_machines}.txt").open()
        data_maker = RegressionDataset(n_tasks=n_tasks, n_machines=n_machines, data_file=data_file)
        train_loader = DataLoader(data_maker, batch_size=len(data_maker))
        inputs, labels = next(iter(train_loader))
        labels = labels.numpy() - torch.sum(inputs[:, :, -1], axis=1).numpy()
        model = LinearRegression()
        model.fit(inputs[:, 1:, :-1].sum(dim=[1, 2]).numpy().reshape(-1, 1), labels)
        regressor = SumRegressor()
        with torch.no_grad():
            regressor.fc.weight = Parameter(Tensor(model.coef_).unsqueeze(0))
            regressor.fc.bias = Parameter(Tensor([model.intercept_]))
            del model
            start = time()
            sum_outputs = np.array(regressor.predict(inputs))[:, -1]
            prediction_time = time() - start
            result = np.mean(np.abs(labels - sum_outputs))
            print(n_tasks, result)
            torch.save(
                regressor.state_dict(),
                f"{Config.OUTPUT_REGRESSION_MODELS}/{regressor}_{n_tasks}_{n_machines}_{(prediction_time / len(data_maker)):.2e}_{result:.1f}.pth",
            )
