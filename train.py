from collections import deque
from itertools import product
from pathlib import Path
from statistics import mean
from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from Config import Config
from ml_models import ConvModel, DenseModel, GRUModel
from ml_models.DataMaker import DataMaker, NoMoreSamplesException


def train(model: nn.Module, n_tasks: int, m_machines: int):
    patience = 1000
    average_size = 100
    batch_size = 16
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_file = Path(f"{Config.TRAINING_DATA_PATH}/{n_tasks}_{n_machines}.txt").open()
    data_maker = DataMaker(n_tasks=n_tasks, n_machines=m_machines, data_file=data_file)
    train_loader = DataLoader(data_maker, batch_size=batch_size)
    losses = deque(maxlen=average_size)
    best_loss = float('inf')
    prediction_time = 0
    try:
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            start = time()
            outputs = model(inputs)
            prediction_time += time() - start
            target = labels.float().unsqueeze(1)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            losses.append((target - outputs).abs().mean().item())
            average = mean(losses)
            print(index, average)
            if index > average_size and average < best_loss:
                best_loss = average
    except NoMoreSamplesException:
        pass
    finally:
        print(best_loss)
        num_predictions = index*batch_size
        torch.save(
            model.state_dict(),
            f'{Config.OUTPUT_MODELS}/{type(model).__name__}_{n_tasks}_{n_machines}_{prediction_time / num_predictions}_{best_loss:.3f}.pth',
        )
        data_file.close()


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    n_machines = 25
    for model_type, n_tasks in product((
        ConvModel,
        DenseModel,
        GRUModel,
    ), range(3, 11)):
        model = model_type(n_tasks=n_tasks, n_machines=n_machines)
        train(model, n_tasks, n_machines)
