import re
from collections import deque
from statistics import fmean
from typing import Type

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from Config import Config
from beam_search.Tree import Tree


def eval_rl(
    n_tasks: int,
    m_machines: int,
    iterations: int,
    model_type: Type[nn.Module],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = dict((int(re.findall(r'\d+', model_path.name)[0]), ((model := model_type(n_tasks, m_machines)), model.load_state_dict(torch.load(model_path)), model.eval(), model.to(device))[0]) for model_path in Config.OUTPUT_RL_MODELS.glob(f'{model_type.__name__}'))
    results = []
    for epoch in range(iterations):
        working_time_matrix = np.random.randint(1, 255, (n_tasks, m_machines))
        tree = Tree(working_time_matrix, models)
        _, state = tree.beam_search()
        results.append(state[-1, -1])


