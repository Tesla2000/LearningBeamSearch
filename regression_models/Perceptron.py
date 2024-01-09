from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn, Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader

from Config import Config
from regression_models.RegressionDataset import RegressionDataset
from regression_models.abstract.BaseRegressor import BaseRegressor


class Perceptron(BaseRegressor):
    learning_rate = 1e-4

    def __init__(self, n_tasks: int, n_machines: int, **_):
        super(Perceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear((n_tasks + 1) * n_machines, 1)

    def predict(self, x):
        x = self.flatten(x)
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
        model.fit(inputs.flatten(start_dim=1).numpy(), labels)
        distances = np.sort(np.abs(labels - model.predict(inputs.flatten(start_dim=1).numpy())))
        for alpha in (.1, .05, .01):
            print(n_tasks, alpha, distances[-int(len(distances) * alpha)])
        plt.hist(distances[:-int(len(distances) * alpha)])
        plt.show()
        # perceptron = Perceptron(n_tasks, n_machines)
        # with torch.no_grad():
        #     perceptron.fc.weight = Parameter(Tensor(model.coef_).unsqueeze(0))
        #     perceptron.fc.bias = Parameter(Tensor([model.intercept_]))
        #     del model
        #     start = time()
        #     line_outputs = np.array(perceptron.predict(Tensor(inputs).float()))[:, -1]
        #     prediction_time = time() - start
        #     result = np.mean(np.abs(labels - line_outputs))
        #     print(n_tasks, result)
        #     torch.save(
        #         perceptron.state_dict(),
        #         f"{Config.OUTPUT_REGRESSION_MODELS}/{perceptron}_{n_tasks}_{n_machines}_{(prediction_time / len(data_maker)):.2e}_{result:.1f}.pth",
        #     )
