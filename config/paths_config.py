from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
RAW_DIR = ARTIFACTS_DIR / "raw"
MODEL_DIR = ARTIFACTS_DIR / "model"


################################ DATA PROCESSING PATHS ################################

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
TRAIN_PATH = RAW_DIR / "titanic_train.csv"
TEST_PATH = RAW_DIR / "titanic_test.csv"

################################ MODEL TRAINING PATHS ################################

MODEL_DIR = ARTIFACTS_DIR / "model"
