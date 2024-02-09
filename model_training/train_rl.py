from collections import deque

import numpy as np
import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from beam_search.Tree import Tree
from model_training.RLDataset import RLDataset
from regression_models.Perceptron import Perceptron


def train_rl(n_tasks: int, m_machines: int, limit: int, models: dict[int, nn.Module] = None):
    training_buffers = dict((key, deque(maxlen=100)) for key in models)
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizers = dict((model, optim.Adam(model.parameters(), lr=model.learning_rate)) for model in models.values())
    for _ in tqdm(range(limit)):
        working_time_matrix = Tensor(np.random.randint(1, 255, (n_tasks, m_machines)))
        tree = Tree(working_time_matrix, models)
        task_order, state = tree.beam_search()
        for tasks in range(min_size, n_tasks + 1):
            if tasks == n_tasks:
                header = np.zeros((1, m_machines))
            else:
                header = state[-tasks - 1].reshape(1, -1)
            data = working_time_matrix[list(task_order[-tasks:])]
            data = np.append(header, data)
            label = state[-1, -1]
            training_buffers[tasks].append((data, label))
        for tasks, model in models.items():
            model.train()
            optimizer = optimizers[model]
            dataset = RLDataset(training_buffers[tasks])
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                target = labels.float().unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()


if __name__ == "__main__":
    n_tasks, m_machines = 10, 25
    min_size = 5
    limit = 10_000
    models = dict((tasks, Perceptron(tasks, m_machines)) for tasks in range(min_size, n_tasks + 1))
    fill_strings = {}
    train_rl(n_tasks, m_machines, limit, models)
