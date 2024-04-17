import numpy as np

from Config import Config
from model_training.RandomNumberGenerator import RandomNumberGenerator


def generate_taillard(generator: RandomNumberGenerator) -> np.array:
    return np.array([[generator.nextInt(1, 100) for _ in range(Config.m_machines)] for _ in range(Config.n_tasks)])
