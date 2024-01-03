from collections import deque
from itertools import product
from pathlib import Path
from statistics import mean
from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from Config import Config
from classification_models.LinearClassifier import LinearClassifier
from classification_models.MinClassifier import MinClassifier
from classification_models.ClassifierDataset import ClassifierDataset
from regression_models.RegressionDataset import RegressionDataset, NoMoreSamplesException
from regression_models.Perceptron import Perceptron
from regression_models.SumRegressor import SumRegressor
from regression_models.WideConvRegressor import WideConvRegressor
from regression_models.WideMultilayerPerceptron import WideMultilayerPerceptron
from regression_models.abstract.BaseRegressor import BaseRegressor


def train_classifier(regressor: BaseRegressor, n_tasks: int, m_machines: int):
    classifier = MinClassifier(regressor, n_tasks)
    average_size = 1000
    batch_size = 16
    learning_rate = classifier.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    data_file = Path(f"{Config.TRAINING_DATA_REGRESSION_PATH}/{n_tasks}_{n_machines}.txt").open()
    data_maker = ClassifierDataset(n_tasks=n_tasks, n_machines=m_machines, data_file=data_file)
    train_loader = DataLoader(data_maker, batch_size=batch_size)
    losses = deque(maxlen=average_size)
    best_loss = float("inf")
    prediction_time = 0
    try:
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            target = labels.float()
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            torch.argmax(outputs, dim=1)
            accuracy = float(torch.sum(labels * torch.nn.functional.one_hot(torch.argmax(outputs, dim=1))) / batch_size)
            losses.append(accuracy)
            average = mean(losses)
            print(index, average)
            if index > average_size and average < best_loss:
                best_loss = average
    except NoMoreSamplesException:
        print(best_loss)
        num_predictions = index * batch_size
        torch.save(
            classifier.state_dict(),
            f"{Config.OUTPUT_CLASSIFIER_MODELS}/{classifier}_{n_tasks}_{n_machines}_{(prediction_time / num_predictions):.2e}_{best_loss:.1f}.pth",
        )
        data_file.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(42)
    n_machines = 25
    for model_type, n_tasks in product(
        (
            # ConvRegressor,
            # MultilayerPerceptron,
            # partial(MultilayerPerceptron, hidden_size=512),
            # GRUModel,
            SumRegressor,
            # Perceptron,
            # WideMultilayerPerceptron,
            # WideConvRegressor,
        ),
        range(4, 11),
    ):
        model: BaseRegressor = model_type(n_tasks=n_tasks - 1, n_machines=n_machines)
        model.load_state_dict(
            torch.load(next(Config.OUTPUT_REGRESSION_MODELS.glob(f"{model}_{n_tasks - 1}_{n_machines}_*"))))
        train_classifier(model, n_tasks, n_machines)
