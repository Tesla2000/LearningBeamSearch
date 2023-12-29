from collections import deque
from copy import deepcopy
from itertools import count
from math import sqrt
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from Config import Config
from ml_models import ConvModel, DenseModel, GRUModel
from ml_models.DataMaker import DataMaker


def train(model: nn.Module, n_tasks: int, m_machines: int):
    patience = 1000
    average_size = 100
    batch_size = 16
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_file = Path(f"data_generation/untitled/data/{n_tasks}_{n_machines}.txt").open()
    data_maker = DataMaker(n_tasks=n_tasks, n_machines=m_machines, length=batch_size, data_file=data_file)
    train_loader = DataLoader(data_maker, batch_size=batch_size)
    losses = deque(maxlen=average_size)
    best_loss = float('inf')
    best_index = 0
    try:
        for epoch in count():
            running_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                target = labels.float().unsqueeze(1)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += sqrt(loss.item())
            losses.append(running_loss)
            average = mean(loss / batch_size for loss in losses)
            print(epoch, average)
            if epoch > average_size and average < best_loss:
                best_loss = average
                best_index = epoch
                model_weights = deepcopy(model.state_dict())
            if best_index + patience < epoch:
                break
    finally:
        print(best_loss)
        torch.save(
            model_weights,
            f'{Config.OUTPUT_MODELS}/{type(model).__name__}_{n_tasks}_{n_machines}_{epoch}_{best_loss:.3f}.pth',
        )
        data_file.close()


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    n_machines = 25
    model = GRUModel(n_machines)
    for n_tasks in range(4, 11):
        # model = ConvModel(n_tasks)
        # model = DenseModel(n_tasks, n_machines)
        train(model, n_tasks, n_machines)
