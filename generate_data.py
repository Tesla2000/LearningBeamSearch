import multiprocessing
import sqlite3
from itertools import count

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from Config import Config
from beam_search.Tree import Tree
from regression_models.Perceptron import Perceptron


def generate_data(output_queue, n_tasks, m_machines, iterations, models: dict[int, nn.Module] = None):
    for _ in range(iterations):
        working_time_matrix = np.random.randint(1, 255, (n_tasks, m_machines))
        tree = Tree(working_time_matrix, models)
        if not models:
            task_order, state = tree.fast_brute_force()
        else:
            task_order, state = tree.beam_search()
        for tasks in range(Config.min_size, n_tasks + 1):
            if tasks == n_tasks:
                continue
            header = state[-tasks - 1].reshape(1, -1)
            data = working_time_matrix[list(task_order[-tasks:])]
            data = np.append(header, data)
            output_queue.put((tasks, list(map(int, data)) + [int(state[-1, -1].item())]))


if __name__ == "__main__":
    conn = sqlite3.connect(Config.DATA_PATH)
    cur = conn.cursor()
    # models = None
    models = dict((tasks, (model := Perceptron(tasks, Config.m_machines),
                           model.load_state_dict(torch.load(next(Config.OUTPUT_REGRESSION_MODELS.glob(
                               f'{type(model).__name__}_{tasks}_{Config.m_machines}*')))), model.eval())[0]) for tasks in
                  range(Config.min_size, 8))
    fill_strings = {}
    for tasks in range(Config.min_size, Config.n_tasks + 1):
        table = Config.table_name(tasks, Config.m_machines)
        cur.execute(
            """CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                {},{},value INTEGER UNSIGNED)""".format(
                table,
                ",".join(
                    map("prev_state_{} INTEGER UNSIGNED".format, range(Config.m_machines))
                ),
                ",".join(
                    map(
                        "worktime_{} TINYINT UNSIGNED".format, range(Config.m_machines * tasks)
                    )
                ),
            )
        )
    conn.commit()
    queue = multiprocessing.Queue()
    processes = []
    for i in range(Config.num_processes):
        process = multiprocessing.Process(target=generate_data, args=(queue, Config.n_tasks, Config.m_machines, Config.n_generated_samples, models))
        process.start()
        processes.append(process)
    counter = iter(tqdm(count()))
    while any(process.is_alive() for process in processes) or not queue.empty():
        while not queue.empty():
            next(counter)
            tasks, data = queue.get()
            table = Config.table_name(tasks, Config.m_machines)
            fill_strings[tasks] = fill_strings.get(
                tasks,
                "INSERT INTO {} ({}, value) VALUES ({})".format(
                    table,
                    ",".join(map("prev_state_{}".format, range(Config.m_machines)))
                    + ","
                    + ",".join(map("worktime_{}".format, range(Config.m_machines * tasks))),
                    ",".join((Config.m_machines * (tasks + 1) + 1) * "?"),
                ),
            )
            cur.execute(fill_strings[tasks], data)
            conn.commit()
    for process in processes:
        process.join()

    conn.close()
