import random
from collections import deque
from functools import partial
from itertools import product
from pathlib import Path
from statistics import mean
from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from Config import Config
from regression_models import GRURegressor, ConvRegressor, MultilayerPerceptron
from regression_models.BranchAndBoundRegressor import BranchAndBoundRegressor
from regression_models.RegressionDataset import RegressionDataset, NoMoreSamplesException
from regression_models.Perceptron import Perceptron
from regression_models.SumRegressor import SumRegressor
from regression_models.WideConvRegressor import WideConvRegressor
from regression_models.WideMultilayerPerceptron import WideMultilayerPerceptron
from regression_models.abstract.BaseRegressor import BaseRegressor


def train_regressor(model: BaseRegressor, n_tasks: int, m_machines: int):
    average_size = 1000
    batch_size = 16
    learning_rate = model.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    if tuple(model.parameters()):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_file = Path(f"{Config.TRAINING_DATA_REGRESSION_PATH}/{n_tasks}_{n_machines}.txt").open()
    data_maker = RegressionDataset(n_tasks=n_tasks, n_machines=m_machines, data_file=data_file)
    train_loader = DataLoader(data_maker, batch_size=batch_size)
    losses = deque(maxlen=average_size)
    best_loss = float("inf")
    prediction_time = 0
    try:
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            target = labels.float().unsqueeze(1)
            if locals().get('model'):
                optimizer.zero_grad()
            start = time()
            outputs = model(inputs)
            prediction_time += time() - start
            if locals().get('model'):
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            losses.append((target - outputs).abs().mean().item())
            average = mean(losses)
            print(index, average)
            if index > average_size and average < best_loss:
                best_loss = average
    except NoMoreSamplesException:
        print(best_loss)
        num_predictions = index * batch_size
        torch.save(
            model.state_dict(),
            f"{Config.OUTPUT_REGRESSION_MODELS}/{model}_{n_tasks}_{n_machines}_{(prediction_time / num_predictions):.2e}_{best_loss:.1f}.pth",
        )
        data_file.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed = 42
    n_machines = 25
    for model_type, n_tasks in product(
        (
            ConvRegressor,
            MultilayerPerceptron,
            partial(MultilayerPerceptron, hidden_size=512),
            SumRegressor,
            Perceptron,
            WideMultilayerPerceptron,
            WideConvRegressor,
        ),
        range(3, 11),
    ):
        model = model_type(n_tasks=n_tasks, n_machines=n_machines)
        train_regressor(model, n_tasks, n_machines)
