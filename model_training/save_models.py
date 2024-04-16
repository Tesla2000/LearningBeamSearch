from pathlib import Path

import torch
from torch import nn

from Config import Config


def save_models(models: dict[int, nn.Module], output_model_path: Path = Config.OUTPUT_RL_MODELS):
    for tasks, model in models.items():
        torch.save(
            model.state_dict(),
            f"{output_model_path}/{type(model).__name__}_{tasks}_{Config.m_machines}.pth",
        )

        # for hidden_state, (n_weights, loss) in model.pareto.items():
        #     for file in Config.OUTPUT_RL_MODELS.glob(
        #         f"{type(model).__name__}_{tasks}_{Config.m_machines}*"
        #     ):
        #         os.remove(file)
        #     dataset = RLDataset(training_buffers[tasks])
        #     torch.save(
        #         model.retrain_hidden_sizes(
        #             hidden_state, Config.criterion, dataset, evaluate=False
        #         ).state_dict(),
        #         f"{Config.OUTPUT_RL_MODELS}/{type(model).__name__}_{tasks}_{Config.m_machines}_{n_weights}_{int(loss)}.pth",
        #     )