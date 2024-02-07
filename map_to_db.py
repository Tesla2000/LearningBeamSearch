import random
import sqlite3
from pathlib import Path

import numpy as np

from Config import Config

if __name__ == '__main__':
    n_machines = 25
    conn = sqlite3.connect(Config.DATA_PATH)
    for n_tasks in range(3, 11):
        print(n_tasks)
        table = f'Samples_{n_tasks}_{n_machines}'
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS {} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        {},{},value INTEGER UNSIGNED)'''.format(table, ','.join(map('prev_state_{} INTEGER UNSIGNED'.format, range(n_machines))),
                                         ','.join(
                                             map('worktime_{} TINYINT UNSIGNED'.format, range(n_machines * n_tasks)))))
        data_file = Path(
            f"{Config.TRAINING_DATA_REGRESSION_PATH}/{n_tasks}_{n_machines}.txt"
        ).open()
        while True:
            prev_state = np.array(
                tuple(map(int, data_file.readline().split()))
            ).reshape(1, -1)
            expected_length = n_tasks * n_machines
            try:
                working_time_matrix = np.array(
                    tuple(map(ord, data_file.read(expected_length)))
                ).reshape((n_tasks, n_machines))
            except ValueError:
                break
            best_value = int(data_file.readline().strip())
            minimal_value = prev_state[0, 0]
            prev_state -= minimal_value
            best_value -= minimal_value
            working_time_matrix = working_time_matrix[
                random.sample(range(len(working_time_matrix)), k=len(working_time_matrix))
            ]
            data = np.append(prev_state[0], working_time_matrix)
            cur.execute('INSERT INTO {} ({}, value) VALUES ({})'.format(table,
                ','.join(map('prev_state_{}'.format, range(n_machines))) + ',' + ','.join(
                    map('worktime_{}'.format, range(n_machines * n_tasks))), ','.join((n_machines * (n_tasks + 1) + 1) * '?')),
                list(data) + [best_value])
        conn.commit()
    conn.close()
