import sqlite3

from torch import Tensor
from torch.utils.data import Dataset

from Config import Config


class RegressionDataset(Dataset):
    def __init__(self, n_tasks: int, m_machines: int):
        self.conn = sqlite3.connect(Config.DATA_PATH)
        self.cur = self.conn.cursor()
        self.table = f"Samples_{n_tasks}_{m_machines}"

    def __len__(self):
        self.cur.execute(f"SELECT COUNT(*) FROM {self.table}")
        return self.cur.fetchone()[0]

    def __getitem__(self, index):
        self.cur.execute(f"SELECT * FROM {self.table} WHERE id = ?", (index + 1,))
        result = self.cur.fetchone()[1:]
        return Tensor(result[:-1]).reshape(1, -1), result[-1]
