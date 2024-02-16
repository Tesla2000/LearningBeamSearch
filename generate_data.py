import multiprocessing
import re
import sqlite3
from itertools import count

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from Config import Config
from beam_search.Tree import Tree
from model_training.database_functions import create_tables, save_sample
from regression_models.Perceptron import Perceptron


def _generate_data(
    output_queue, n_tasks, m_machines, iterations, models: dict[int, nn.Module] = None
):
    for _ in range(iterations):
        working_time_matrix = np.random.randint(1, 255, (n_tasks, m_machines))
        tree = Tree(working_time_matrix, models)
        if not models:
            task_order, state = tree.fast_brute_force()
        else:
            task_order, state = tree.beam_search()
        for tasks in range(Config.min_saving_size, n_tasks):
            header = state[-tasks - 1].reshape(1, -1)
            data = working_time_matrix[list(task_order[-tasks:])]
            data = np.append(header, data)
            output_queue.put(
                (tasks, list(map(int, data)) + [int(state[-1, -1].item())])
            )


def generate_data(n_tasks: int):
    conn = sqlite3.connect(Config.DATA_PATH)
    cur = conn.cursor()
    model_type = Perceptron
    models = dict(
        (
            int(re.findall(r"\d+", model_path.name)[0]),
            (
                model := model_type(
                    int(re.findall(r"\d+", model_path.name)[0]), Config.m_machines
                ),
                model.load_state_dict(torch.load(model_path)),
                model.eval(),
            )[0],
        )
        for model_path in Config.OUTPUT_REGRESSION_MODELS.glob(
            f"{model_type.__name__}_*"
        )
    )
    fill_strings = {}
    create_tables(conn, cur)
    queue = multiprocessing.Queue()
    processes = []
    for i in range(Config.num_processes):
        process = multiprocessing.Process(
            target=_generate_data,
            args=(
                queue,
                n_tasks,
                Config.m_machines,
                Config.n_generated_samples,
                models,
            ),
        )
        process.start()
        processes.append(process)
    counter = iter(tqdm(count()))
    while any(process.is_alive() for process in processes) or not queue.empty():
        while not queue.empty():
            next(counter)
            tasks, data = queue.get()
            save_sample(tasks, data, fill_strings, conn, cur)
    for process in processes:
        process.join()

    conn.close()


if __name__ == "__main__":
    generate_data(Config.n_tasks)
