import re
import sqlite3
from itertools import count

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from time import time

from Config import Config
from beam_search.Tree import Tree
from model_training.database_functions import create_tables, save_sample
from regression_models.Perceptron import GenPerceptron


def _generate_data(
    n_tasks, m_machines, models: dict[int, nn.Module] = None
):
    working_time_matrix = np.random.randint(1, 255, (n_tasks, m_machines))
    tree = Tree(working_time_matrix, models, verbose=False)
    if not models:
        task_order, state = tree.fast_brute_force()
    else:
        task_order, state = tree.beam_search(Config.minimal_beta)
    for tasks in range(Config.min_size, n_tasks):
        header = state[-tasks - 1].reshape(1, -1)
        data = working_time_matrix[list(task_order[-tasks:])]
        data = np.append(header, data)
        yield tasks, list(map(int, data)) + [int(state[-1, -1].item())]


def generate_data():
    conn = sqlite3.connect(Config.DATA_PATH)
    cur = conn.cursor()
    model_type = GenPerceptron
    models = dict(
        (
            int(re.findall(r"\d+", model_path.name)[0]),
            (
                model := model_type(
                    int(re.findall(r"\d+", model_path.name)[0]), Config.m_machines
                ),
                model.load_state_dict(torch.load(model_path)),
                model.to(Config.device),
                model.eval(),
            )[0],
        )
        for model_path in Config.OUTPUT_RL_MODELS.glob(
            f"{model_type.__name__}_*"
        )
    )
    fill_strings = {}
    create_tables(conn, cur)
    start = time()
    for _ in tqdm(count()):
        if start + Config.data_generation_time < time():
            break
        for tasks, data in _generate_data(Config.n_tasks, Config.m_machines, models):
            save_sample(tasks, data, fill_strings, conn, cur)

    conn.close()


if __name__ == "__main__":
    generate_data()
