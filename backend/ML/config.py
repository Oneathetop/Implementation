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

# --------------------------------------------------
# Feature Names

FEATURE_NAMES = [

    # URL Structure
    "url_length",
    "domain_length",
    "path_length",
    "slash_count",
    "double_slash_in_path",

    # Domain
    "suspicious_file_extension",
    "subdomain_count",
    "query_parameter_count",
    "https",
    "ip_address",

    # Character Statistics
    "dot_count",
    "digit_count",
    "max_consecutive_digits",
    "hyphen_count",
    "special_character_count",
    "at_symbol_presence",
    "suspicious_character_ratio",
    "url_encoding_count",
    "entropy",
    "consecutive_special_character_count",

    # Phishing Intelligence
    "keyword_count",
    "brand_keyword_count",
    "url_shortener",
    "port_number",
    "suspicious_tld",
    "fragment_present",
]