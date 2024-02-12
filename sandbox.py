import sqlite3

from Config import Config

if __name__ == '__main__':
    conn = sqlite3.connect(Config.DATA_PATH)
    cur = conn.cursor()
    for n_tasks in range(7, 10):
        cur.execute(
            """DROP TABLE IF EXISTS {}""".format(Config.table_name(n_tasks, 25))
        )
        conn.commit()
    conn.close()
