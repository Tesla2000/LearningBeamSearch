from copy import deepcopy
from itertools import count
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


def train_regressor(model: BaseRegressor, n_tasks: int, m_machines: int):
    if tuple(Config.OUTPUT_REGRESSION_MODELS.glob(f"{model}_{n_tasks}_{m_machines}*")):
        return
    results = []
    batch_size = 32
    test_percentage = 0.2
    learning_rate = model.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=Config.gamma)
    best_result = float("inf")
    for epoch in count():
        dataset = RegressionDataset(n_tasks=n_tasks, m_machines=m_machines)
        train_set, val_set = torch.utils.data.random_split(
            dataset,
            [
                len(dataset) - int(test_percentage * len(dataset)),
                int(test_percentage * len(dataset)),
            ],
        )
        del dataset
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=len(val_set))
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            target = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        inputs, labels = next(iter(val_loader))
        inputs, labels = inputs.to(device), labels.to(device)
        target = labels.float().unsqueeze(1)
        model.eval()
        start = time()
        outputs = model(inputs)
        prediction_time = (time() - start) / len(outputs)
        result = (target - outputs).abs().mean()
        results.append(result)
        best_result = min(results)
        best_result_epoch = results.index(best_result)
        if epoch >= best_result_epoch + Config.patience:
            torch.save(
                state_dict,
                f"{Config.OUTPUT_REGRESSION_MODELS}/{model}_{n_tasks}_{m_machines}_{prediction_time:.2e}_{best_result:.1f}.pth",
            )
            return
        if best_result_epoch == epoch:
            state_dict = deepcopy(model.state_dict())
        print(epoch, result.item())
