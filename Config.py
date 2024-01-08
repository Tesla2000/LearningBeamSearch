from pathlib import Path


class Config:
    ROOT = Path(__file__).parent
    OUTPUT_REGRESSION_MODELS = ROOT / "output_regression_models"
    OUTPUT_CLASSIFIER_MODELS = ROOT / "output_classifier_models"
    TRAINING_DATA_REGRESSION_PATH = ROOT / Path("data_generation/untitled/training_data_regression")
    TRAINING_DATA_CLASSIFICATION_PATH = ROOT / Path("data_generation/untitled/training_data_classification")
