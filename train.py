from collections import deque
from itertools import count
from math import sqrt
from statistics import mean

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ml_models.ConvModel import ConvModel
from ml_models.DataMaker import DataMaker


def train(model: nn.Module, n_tasks: int, m_machines: int, rows: int):
    patience = 200
    average_size = 100
    batch_size = 16
    learning_rate = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    data_maker = DataMaker(n_tasks=n_tasks, m_machines=m_machines, rows=rows, length=batch_size)
    train_loader = DataLoader(data_maker, batch_size=batch_size, shuffle=True)
    losses = deque(maxlen=average_size)
    best_loss = 1e10
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
            running_loss += loss.item()
        losses.append(running_loss)
        average = mean(losses) / batch_size
        print(epoch, sqrt(average))
        if average < best_loss:
            best_loss = average
            best_index = epoch
        if best_index + patience < epoch:
            print(best_loss)
            break


if __name__ == '__main__':
    n_tasks = 7
    m_machines = 10
    rows = 5
    model = ConvModel(rows)
    train(model, n_tasks, m_machines, rows)
