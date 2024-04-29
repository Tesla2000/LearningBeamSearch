from pathlib import Path

import torch
from torch import nn

from Config import Config


def save_models(models: dict[int, nn.Module], output_model_path: Path = Config.OUTPUT_RL_MODELS):
    for tasks, model in models.items():
        torch.save(
            model.state_dict(),
            f"{output_model_path}/{model.name}_{tasks}_{Config.m_machines}.pth",
        )