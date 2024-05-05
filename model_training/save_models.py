from pathlib import Path
from statistics import fmean
from typing import Sequence

import torch
from torch import nn

from Config import Config


def save_models(models: dict[int, nn.Module], output_model_path: Path = Config.OUTPUT_RL_MODELS):
    for tasks, model in models.items():
        torch.save(
            model.state_dict(),
            f"{output_model_path}/{model.name}_{tasks}_{Config.m_machines}.pth",
        )


def save_genetic_models(models: Sequence[nn.Module], output_model_path: Path = Config.OUTPUT_RL_MODELS):
    for model in models:
        torch.save(
            model.state_dict(),
            f"{output_model_path}/{model.name}_{fmean(model.correctness_of_predictions)}.pth",
        )
