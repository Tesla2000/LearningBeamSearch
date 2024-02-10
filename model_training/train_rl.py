from collections import deque
from statistics import fmean
from time import time
from typing import IO

import numpy as np
import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader

from Config import Config
from beam_search.Tree import Tree
from model_training.RLDataset import RLDataset


def train_rl(
    n_tasks: int,
    m_machines: int,
    iterations: int,
    min_size: int,
    models: dict[int, nn.Module] = None,
    output_file: IO = None,
):
    training_buffers = dict((key, deque(maxlen=100)) for key in models)
    results = []
    buffered_results = deque(maxlen=Config.results_average_size)
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizers = dict(
        (model, optim.Adam(model.parameters(), lr=model.learning_rate))
        for model in models.values()
    )
    start = time()
    for epoch in range(iterations):
        working_time_matrix = Tensor(np.random.randint(1, 255, (n_tasks, m_machines)))
        tree = Tree(working_time_matrix, models, Config.beta)
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
        buffered_results.append(label.item())
        if epoch > Config.minimal_counting_epoch_number:
            results.append(fmean(buffered_results))
            output_file.write(f"{int(time() - start)},{results[-1]:.2f}\n")
            print(epoch, results[-1])
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
    save_models(models, results[-1])


def save_models(models: dict[int, nn.Module], best_result: int):
    for tasks, model in models.items():
        torch.save(
            model.state_dict(),
            f"{Config.OUTPUT_REGRESSION_MODELS}/{model}_{tasks}_{Config.m_machines}_{best_result}.pth",
        )