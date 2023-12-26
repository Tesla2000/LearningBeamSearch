from collections import deque
from itertools import count
from math import sqrt
from statistics import mean

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ml_models.ConvModel import ConvModel
from ml_models.DataMaker import DataMaker
from ml_models.LSTMModel import LSTMModel


def train(model: nn.Module, n_tasks: int, m_machines: int, rows: int):
    patience = 1000
    average_size = 100
    batch_size = 16
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_maker = DataMaker(n_tasks=n_tasks, m_machines=m_machines, rows=rows, length=batch_size)
    train_loader = DataLoader(data_maker, batch_size=batch_size, shuffle=True)
    losses = deque(maxlen=average_size)
    best_loss = float('inf')
    best_index = 0
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
            model_weights = model.state_dict()
        if best_index + patience < epoch:
            print(best_loss)
            torch.save(
                model_weights,
                f'output_models/{rows}_{best_loss:.3f}.pth',
            )
            break


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    n_tasks = 7
    m_machines = 10
    rows = 5
    # model = ConvModel(rows)
    model = LSTMModel(m_machines)
    train(model, n_tasks, m_machines, rows)
