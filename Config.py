from functools import partial
from pathlib import Path

from regression_models import MultilayerPerceptron, ConvRegressor
from regression_models.Perceptron import Perceptron
from regression_models.WideMultilayerPerceptron import WideMultilayerPerceptron


class Config:
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
    DATA_PATH = ROOT / Path("data.db")
    RL_DATA_PATH = ROOT / Path("rl_data.db")

    table_name = "Samples_{}_{}".format

    patience = 2
    min_model_size = 5
    min_saving_size = 7
    n_generated_samples = 50_000
    num_processes = 4

    model_types = [
        # WideMultilayerPerceptron,
        # MultilayerPerceptron,
        # Perceptron,
        ConvRegressor,
    ]
    universal_model_types = tuple()

    n_tasks, m_machines = 50, 25
    min_size = 4
    train_time = 12 * 3600
    minimal_counting_epoch_number = 500
    results_average_size = 100
    training_buffer_size = 5000
    beta = dict((tasks, 1000) for tasks in range(n_tasks + 1))
    minimal_beta = dict((tasks, 50) for tasks in range(n_tasks + 1))
    beta_attrition = 0.998
    gamma = 0.999
    eval_iterations = 500
    save_interval = 10
    max_status_length = 2000


# from regression_models.UniversalEfficientNet import UniversalEfficientNetAnySize, UniversalEfficientNetMaxSize
#
# Config.universal_model_types = (
#     UniversalEfficientNetAnySize,
#     UniversalEfficientNetMaxSize,
# )
