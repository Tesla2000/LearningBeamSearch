import re
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


def test_rl(
    n_tasks: int,
    m_machines: int,
    iterations: int,
    min_size: int,
    model_type: nn.Module,
):
    results = []
    buffered_results = deque(maxlen=Config.results_average_size)
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    models = dict((int(re.findall(r'\d+', model_path.name)[0]), model_path) for model_path in Config.OUTPUT_RL_MODELS.glob(f'{model_type.__name__}'))
    start = time()
    for epoch in range(iterations):
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
        buffered_results.append(label.item())
        results.append(fmean(buffered_results))
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
            f"{Config.OUTPUT_RL_MODELS}/{model}_{tasks}_{Config.m_machines}_{best_result}.pth",
        )
