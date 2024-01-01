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
from ml_models import MultilayerPerceptron
from ml_models.DataMaker import DataMaker, NoMoreSamplesException
from ml_models.abstract.BaseModel import BaseModel
from ml_models.abstract.DropoutModel import DropoutModel


def train(model: BaseModel, n_tasks: int, m_machines: int):
    average_size = 1000
    batch_size = 16
    learning_rate = model.learning_rate
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
            target = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            if isinstance(model, DropoutModel):
                model.eval()
                outputs = model(inputs)
                losses.append((target - outputs).abs().mean().item())
                model.train()
            start = time()
            outputs = model(inputs)
            prediction_time += time() - start
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            if not isinstance(model, DropoutModel):
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
        # ConvModel,
        MultilayerPerceptron,
        # GRUModel,
        # SumModel,
        # Perceptron,
    ), range(3, 11)):
        model = model_type(n_tasks=n_tasks, n_machines=n_machines)
        train(model, n_tasks, n_machines)
