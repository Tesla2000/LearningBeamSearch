from pathlib import Path
from statistics import fmean
from typing import Sequence

import torch
from torch import nn

from Config import Config
from models.abstract.BaseRegressor import BaseRegressor


def save_models(models: dict[int, BaseRegressor], output_model_path: Path = Config.OUTPUT_RL_MODELS):
    for tasks, model in models.items():
        folder = output_model_path / str(Config.n_tasks) / str(Config.m_machines)
        folder.mkdir(exist_ok=True, parents=True)
        torch.save(
            model.state_dict(),
            f"{folder}/{model.name}_{tasks}.pth",
        )


def save_genetic_models(models: Sequence[nn.Module], output_model_path: Path = Config.OUTPUT_RL_MODELS):
    for model in models:
        torch.save(
            model.state_dict(),
            f"{output_model_path}/{model.name}_{fmean(model.correctness_of_predictions)}.pth",
        )
