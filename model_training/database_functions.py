from sqlite3 import Connection, Cursor

from Config import Config


def save_sample(
    tasks: int, data, fill_strings: dict[int, str], conn: Connection, cur: Cursor
):
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


def create_tables(conn: Connection, cur: Cursor):
    for tasks in range(Config.min_model_size, Config.n_tasks + 1):
        table = Config.table_name(tasks, Config.m_machines)
        cur.execute(
            """CREATE TABLE IF NOT EXISTS {} (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                {},{},value INTEGER UNSIGNED)""".format(
                table,
                ",".join(
                    map(
                        "prev_state_{} INTEGER UNSIGNED".format,
                        range(Config.m_machines),
                    )
                ),
                ",".join(
                    map(
                        "worktime_{} TINYINT UNSIGNED".format,
                        range(Config.m_machines * tasks),
                    )
                ),
            )
        )
    conn.commit()
