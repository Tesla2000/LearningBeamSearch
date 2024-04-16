import operator
from collections import deque
from functools import reduce
from itertools import count
from statistics import fmean
from time import time
from typing import IO

from torch import optim, Tensor
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from Config import Config
from beam_search.RecurrentTree import RecurrentTree
from model_training.RLDataset import RLDataset
from model_training.RandomNumberGenerator import RandomNumberGenerator
from model_training.generate_taillard import generate_taillard
from model_training.save_models import save_models
from regression_models.GeneticRegressor import GeneticRegressor
from regression_models.RecurrentModel import RecurrentModel


def recurrent_model(
    generator: RandomNumberGenerator,
    models: dict[int, RecurrentModel] = None,
    output_file: IO = None,
    **_,
):
    _, model = models.popitem()
    training_buffer = deque(maxlen=Config.train_buffer_size)
    results = []
    buffered_results = deque(maxlen=Config.results_average_size)
    optimizer = optim.Adam(
                (
                    model.best_model if isinstance(model, GeneticRegressor) else model
                ).parameters(),
                lr=getattr(model, "learning_rate", 1e-5),
            )
    scheduler = ExponentialLR(optimizer, Config.gamma)
    start = time()
    for epoch in count(1):
        if start + Config.train_time < time():
            break
        working_time_matrix = generate_taillard(generator)
        tree = RecurrentTree(working_time_matrix)
        task_order, state = tree.beam_search(model, Config.beta)
        label = state[-1, -1]
        training_buffer.append((working_time_matrix, task_order, label))
        buffered_results.append(label.item())
        results.append(fmean(buffered_results))
        output_file.write(f"{int(time() - start)},{results[-1]:.2f}\n")
        print(epoch, results[-1])
# ========================================================================================================
        model.train()
        dataset = RLDataset(training_buffer)
        train_loader = DataLoader(dataset)
        for working_time_matrix, task_order, label in train_loader:
            working_time_matrix, task_order, label = working_time_matrix.squeeze(0).to(Config.device), task_order.to(Config.device), label.to(Config.device)
            label = label.float().unsqueeze(1)
            task_order = task_order.flatten()
            hn = Tensor(working_time_matrix).flatten().unsqueeze(0).to(Config.device).float()
            losses = []
            for task in task_order:
                optimizer.zero_grad()
                outputs, hn = model(Tensor(working_time_matrix[task]).unsqueeze(0).to(Config.device), hn)
                losses.append(Config.criterion(outputs, label))
            loss = reduce(operator.add, losses)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % Config.save_interval == 0:
            save_models(models)
    save_models(models)
