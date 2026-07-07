from pathlib import Path

# --------------------------------------------------
# Project Directories

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_RAW = BASE_DIR / "datasets" / "raw"
DATASET_PROCESSED = BASE_DIR / "datasets" / "processed"

MODEL_DIR = BASE_DIR / "models"

LOG_DIR = BASE_DIR / "logs"

# --------------------------------------------------
# Datasets

RAW_DATASET = DATASET_RAW / "PhiUSIIL_Phishing_URL_Dataset.xlsx"

CLEAN_DATASET = DATASET_PROCESSED / "clean_urls.csv"

# --------------------------------------------------
# ML Parameters

RANDOM_STATE = 42

TEST_SIZE = 0.20