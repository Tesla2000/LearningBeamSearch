from collections import deque
from statistics import mean
from time import time

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from Config import Config
from model_training.RegressionDataset import (
    RegressionDataset,
)
from regression_models.abstract.BaseRegressor import BaseRegressor


def train_regressor(model: BaseRegressor, n_tasks: int, n_machines: int):
    average_size = 1000
    batch_size = 16
    learning_rate = model.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    if tuple(model.parameters()):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=.1)
    best_loss = float("inf")
    prediction_time = 0
    losses = deque(maxlen=average_size)
    for epoch in range(3):
        data_maker = RegressionDataset(
            n_tasks=n_tasks, n_machines=n_machines
        )
        train_loader = DataLoader(data_maker, batch_size=batch_size)
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            target = labels.float().unsqueeze(1)
            if locals().get("model"):
                optimizer.zero_grad()
            start = time()
            outputs = model(inputs)
            prediction_time += time() - start
            if locals().get("model"):
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            losses.append((target - outputs).abs().mean().item())
            average = mean(losses)
            print(index, average)
        scheduler.step()
    print(best_loss)
    num_predictions = index * batch_size
    torch.save(
        model.state_dict(),
        f"{Config.OUTPUT_REGRESSION_MODELS}/{model}_{n_tasks}_{n_machines}_{(prediction_time / num_predictions):.2e}_{best_loss:.1f}.pth",
    )

