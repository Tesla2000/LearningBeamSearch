from pathlib import Path
from time import time
from numpy.linalg import lstsq
import numpy as np
import torch
from torch.utils.data import DataLoader

from Config import Config
from regression_models.RegressionDataset import RegressionDataset

if __name__ == '__main__':
    n_machines = 25
    for n_tasks in range(3, 11):
        data_file = Path(f"{Config.TRAINING_DATA_REGRESSION_PATH}/{n_tasks}_{n_machines}.txt").open()
        data_maker = RegressionDataset(n_tasks=n_tasks, n_machines=n_machines, data_file=data_file)
        train_loader = DataLoader(data_maker, batch_size=len(data_maker))
        inputs, labels = next(iter(train_loader))
        labels = labels.numpy()
        inputs = inputs.flatten(start_dim=1).numpy()
        mod_input = np.empty((inputs.shape[0], inputs.shape[1] + 1))
        mod_input[:, :inputs.shape[1]] = inputs
        mod_input[:, -1] = np.ones_like(mod_input[:, -1])
        w = lstsq(mod_input, labels)[0]
        output = mod_input.dot(w)
        results = np.mean(np.abs(labels - output))


