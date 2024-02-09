from pathlib import Path


class Config:
    ROOT = Path(__file__).parent
    OUTPUT_REGRESSION_MODELS = ROOT / "output_regression_models"
    OUTPUT_REGRESSION_MODELS.mkdir(exist_ok=True)
    DATA_PATH = ROOT / Path("data.db")
