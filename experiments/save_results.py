from typing import Type

from Config import Config
from models.abstract.BaseRegressor import BaseRegressor


def save_results(model_type: Type[BaseRegressor], number: int, beta: bool, results: list):
    folder = Config.OUTPUT_RL_RESULTS.joinpath(str(Config.n_tasks)).joinpath(str(Config.m_machines)).joinpath("beta" if beta else "time").joinpath(str(number))
    folder.mkdir(exist_ok=True, parents=True)
    folder.joinpath(model_type.__name__).write_text(str(results))
