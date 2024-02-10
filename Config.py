from collections import defaultdict
from pathlib import Path


class Config:
    ROOT = Path(__file__).parent
    OUTPUT_REGRESSION_MODELS = ROOT / "output_regression_models"
    OUTPUT_REGRESSION_MODELS.mkdir(exist_ok=True)
    MODEL_RESULTS = ROOT / "model_train_log"
    MODEL_RESULTS.mkdir(exist_ok=True)
    DATA_PATH = ROOT / Path("data.db")
    minimal_counting_epoch_number = 500

    beta = defaultdict(lambda: 50)
    min_size = 5
    n_tasks, m_machines = 7, 25
    n_generated_samples = 100_000
    num_processes = 3
