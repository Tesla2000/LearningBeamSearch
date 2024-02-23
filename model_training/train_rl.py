import random
import sqlite3
from collections import deque
from itertools import count
from statistics import fmean
from time import time
from typing import IO

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from Config import Config
from beam_search.Tree import Tree
from model_training.RLDataset import RLDataset
from model_training.database_functions import create_tables, save_sample
from regression_models.RecurrentModel import RecurrentModel


def train_rl(
    n_tasks: int,
    m_machines: int,
    min_size: int,
    models: dict[int, nn.Module] = None,
    output_file: IO = None,
):
    # fill_strings = {}
    conn = sqlite3.connect(Config.RL_DATA_PATH)
    cur = conn.cursor()
    create_tables(conn, cur)
    training_buffers = dict((key, deque(maxlen=100)) for key in models)
    results = []
    buffered_results = deque(maxlen=Config.results_average_size)
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizers = dict(
        (model, optim.Adam(model.parameters(), lr=getattr(model, 'learning_rate', 1e-5)))
        for model in models.values()
    )
    schedulers = dict(
        (optimizer, ExponentialLR(optimizer, Config.gamma))
        for optimizer in optimizers.values()
    )
    recurrent = any(isinstance(model, RecurrentModel) for model in models.values())
    start = time()
    for epoch in count(1):
        if start + Config.train_time < time():
            break
        working_time_matrix = np.random.randint(1, 255, (n_tasks, m_machines))
        if recurrent:
            random.choice(tuple(models.values())).fill_state(working_time_matrix)
        tree = Tree(working_time_matrix, models)
        task_order, state = tree.beam_search(Config.beta)
        # for tasks in range(Config.min_saving_size, n_tasks):
        #     header = state[-tasks - 1].reshape(1, -1)
            # data = working_time_matrix[list(task_order[-tasks:])]
            # data = np.append(header, data)
            # data = list(map(int, data)) + [int(state[-1, -1].item())]
            # save_sample(tasks, data, fill_strings, conn, cur)
        for tasks in range(min_size, n_tasks + 1):
            if tasks == n_tasks:
                header = np.zeros((1, m_machines))
            else:
                header = state[-tasks - 1].reshape(1, -1)
            data = working_time_matrix[list(task_order[-tasks:])]
            data = np.append(header, data, axis=0)
            label = state[-1, -1]
            training_buffers[tasks].append((data, label))
        buffered_results.append(label.item())
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
                outputs = model(inputs.float())
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            schedulers[optimizer].step()
            for key in Config.beta:
                Config.beta[key] *= Config.beta_attrition
            schedulers[optimizer].step()
        if epoch % Config.save_interval == 0:
            save_models(models)
    save_models(models)


def save_models(models: dict[int, nn.Module]):
    saved = set()
    for tasks, model in models.items():
        if id(model) in saved:
            continue
        saved.add(id(model))
        torch.save(
            model.state_dict(),
            f"{Config.OUTPUT_RL_MODELS}/{type(model).__name__}_{tasks}_{Config.m_machines}.pth",
        )
