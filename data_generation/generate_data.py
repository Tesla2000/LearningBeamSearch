import sqlite3

import numpy as np
from tqdm import tqdm

from Config import Config
from beam_search.Tree import Tree


def generate_data(n_tasks, m_machines, limit):
    for _ in tqdm(range(limit)):
        working_time_matrix = np.random.randint(1, 255, (n_tasks, m_machines))
        tree = Tree(working_time_matrix)
        result = tree.fast_brute_force()
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
                header = result.state[[result.tasks[-tasks - 1]]]
            value = result.value - header[0, 0]
            header -= header[0, 0]
            data = working_time_matrix[list(result.tasks[-tasks:])]
            data = np.append(header, data)
            cur.execute(fill_strings[tasks], list(data) + [value])
        conn.commit()


if __name__ == "__main__":
    n_tasks, m_machines = 3, 25
    limit = 100_000
    conn = sqlite3.connect(Config.DATA_PATH)
    cur = conn.cursor()
    min_size = 3
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
    generate_data(n_tasks, m_machines, limit)
    conn.close()
