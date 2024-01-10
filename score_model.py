from pathlib import Path

from torch.utils.data import DataLoader

from Config import Config
from regression_models.RegressionDataset import RegressionDataset

if __name__ == "__main__":
    n_machines = 25
    for n_tasks in range(3, 11):
        data_file = Path(
            f"{Config.TRAINING_DATA_REGRESSION_PATH}/{n_tasks}_{n_machines}.txt"
        ).open()
        data_maker = RegressionDataset(
            n_tasks=n_tasks, n_machines=n_machines, data_file=data_file
        )
        train_loader = DataLoader(data_maker, batch_size=len(data_maker))
        inputs, labels = next(iter(train_loader))
