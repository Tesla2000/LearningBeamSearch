from pathlib import Path

import torch

from regression_models import MultilayerPerceptron
from regression_models.Perceptron import Perceptron
from regression_models.WideMultilayerPerceptron import WideMultilayerPerceptron


class GeneticConfig:
    gen_train_epochs = 3
    n_genetic_samples = 10
    n_genetic_models = 20
    retrain_rate = .2
    size_penalty = 100


class Config(GeneticConfig):
    train = True

    ROOT = Path(__file__).parent
    OUTPUT_REGRESSION_MODELS = ROOT / "output_regression_models"
    OUTPUT_REGRESSION_MODELS.mkdir(exist_ok=True)
    OUTPUT_RL_MODELS = ROOT / "output_rl_models"
    OUTPUT_RL_MODELS.mkdir(exist_ok=True)
    OUTPUT_RL_RESULTS = ROOT / "output_rl_results"
    OUTPUT_RL_RESULTS.mkdir(exist_ok=True)
    MODEL_RESULTS = ROOT / "model_train_log"
    MODEL_RESULTS.mkdir(exist_ok=True)
    PLOTS = ROOT / "plots"
    PLOTS.mkdir(exist_ok=True)
    DATA_PATH = ROOT / Path("data.db")
    RL_DATA_PATH = ROOT / Path("rl_data.db")

    table_name = "Samples_{}_{}".format

    patience = 2
    min_model_size = 5
    min_saving_size = 7
    n_generated_samples = 50_000
    num_processes = 4

    universal_model_types = tuple()
    recurrent_model_types = tuple()
    model_types = tuple()

    n_tasks, m_machines = 50, 25
    min_size = 4
    train_time = 12 * 3600
    minimal_counting_time = 1800
    results_average_size = 100
    train_buffer_size = 100
    beta = dict((tasks, 1000) for tasks in range(n_tasks + 1))
    minimal_beta = dict((tasks, 50) for tasks in range(n_tasks + 1))
    beta_attrition = 0.99
    gamma = 0.999
    eval_iterations = 500
    save_interval = 10
    max_status_length = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# from regression_models.UniversalEfficientNet import UniversalEfficientNetAnySize, UniversalEfficientNetMaxSize
# from regression_models.EncodingPerceptron import EncodingPerceptron
from regression_models.ZeroPaddedPerceptron import ZeroPaddedPerceptron
from regression_models.GeneticRegressor import GeneticRegressor

Config.model_types = (
    # WideMultilayerPerceptron,
    # MultilayerPerceptron,
    # Perceptron,
    GeneticRegressor,
)
Config.universal_model_types = (
    # UniversalEfficientNetAnySize,
    # UniversalEfficientNetMaxSize,
    # EncodingPerceptron,
    # ZeroPaddedPerceptron,
)
# from regression_models.RecurrentModel import RecurrentModel
#
# Config.recurrent_model_types = (
#     lambda: RecurrentModel(_encoder),
# )
