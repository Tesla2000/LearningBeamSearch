import sqlite3

import numpy as np
import torch
from torch import nn, Tensor
from tqdm import tqdm

from Config import Config
from beam_search.Tree import Tree
from regression_models.Perceptron import Perceptron


def generate_data(n_tasks, m_machines, limit, models: dict[int, nn.Module] = None):
    for _ in tqdm(range(limit)):
        working_time_matrix = Tensor(np.random.randint(1, 255, (n_tasks, m_machines)))
        tree = Tree(working_time_matrix, models)
        if not models:
            task_order, state = tree.fast_brute_force()
        else:
            task_order, state = tree.beam_search()
        for tasks in range(min_size, n_tasks + 1):
            table = f"Samples_{tasks}_{m_machines}"
            fill_strings[tasks] = fill_strings.get(
                tasks,
                "INSERT INTO {} ({}, value) VALUES ({})".format(
                    table,
                    ",".join(map("prev_state_{}".format, range(m_machines)))
                    + ","
                    + ",".join(map("worktime_{}".format, range(m_machines * tasks))),
                    ",".join((m_machines * (tasks + 1) + 1) * "?"),
                ),
            )
            if tasks == n_tasks:
                header = np.zeros((1, m_machines))
            else:
                header = state[-tasks - 1].reshape(1, -1)
            data = working_time_matrix[list(task_order[-tasks:])]
            data = np.append(header, data)
            cur.execute(fill_strings[tasks], list(data) + [state[-1, -1]])
        conn.commit()


if __name__ == "__main__":
    n_tasks, m_machines = 7, 25
    limit = 100_000
    conn = sqlite3.connect(Config.DATA_PATH)
    cur = conn.cursor()
    min_size = 3
    models = None
    # models = dict((tasks, (model := Perceptron(tasks, m_machines),
    #                        model.load_state_dict(torch.load(next(Config.OUTPUT_REGRESSION_MODELS.glob(
    #                            f'{type(model).__name__}_{tasks}_{m_machines}*')))), model.eval())[0]) for tasks in
    #               range(min_size, 7))
    fill_strings = {}
    for tasks in range(min_size, n_tasks + 1):
        table = f"Samples_{tasks}_{m_machines}"
        cur.execute(
            """CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                {},{},value INTEGER UNSIGNED)""".format(
                table,
                ",".join(
                    map("prev_state_{} INTEGER UNSIGNED".format, range(m_machines))
                ),
                ",".join(
                    map(
                        "worktime_{} TINYINT UNSIGNED".format, range(m_machines * tasks)
                    )
                ),
            )
        )
    conn.commit()
    generate_data(n_tasks, m_machines, limit, models)
    conn.close()
