from collections import defaultdict
from pathlib import Path

from regression_models import MultilayerPerceptron
from regression_models.Perceptron import Perceptron


class Config:
    ROOT = Path(__file__).parent
    OUTPUT_REGRESSION_MODELS = ROOT / "output_regression_models"
    OUTPUT_REGRESSION_MODELS.mkdir(exist_ok=True)
    OUTPUT_RL_MODELS = ROOT / "output_rl_models"
    OUTPUT_RL_MODELS.mkdir(exist_ok=True)
    MODEL_RESULTS = ROOT / "model_train_log"
    MODEL_RESULTS.mkdir(exist_ok=True)
    DATA_PATH = ROOT / Path("data.db")

    table_name = "Samples_{}_{}".format

    patience = 2
    min_model_size = 5
    min_saving_size = 7
    n_generated_samples = 50_000
    num_processes = 4

    model_types = (
        Perceptron,
        MultilayerPerceptron,
    )

    n_tasks, m_machines = 20, 25
    min_size = 4
    iterations = 2000
    minimal_counting_epoch_number = 500
    results_average_size = 100
    training_buffer_size = 5000
    beta = defaultdict(lambda: 50)
    gamma = 0.999
