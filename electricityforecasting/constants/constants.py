from pathlib import Path

# ------------- CORE PATHS -------------
ARTIFACTS_ROOT = Path("artifacts")
RAW_DATA_DIR = ARTIFACTS_ROOT / "data_ingestion"
VALIDATED_DATA_DIR = ARTIFACTS_ROOT / "data_validation"
TRANSFORMED_DATA_DIR = ARTIFACTS_ROOT / "data_transformation"
FORECASTING_RESULTS_DIR = Path("forecasting_results")

# ------------- FILE NAMES -------------
RAW_DF_FILE = RAW_DATA_DIR / "raw_data.parquet"
IMPUTATION_LOG_FILE = VALIDATED_DATA_DIR / "imputation_log.csv"
CLEAN_DF_FILE = VALIDATED_DATA_DIR / "clean_data.parquet"

# ------------- DATA PROCESSING -------------
EXCLUDE_STATES = ("Pondy", "Tripura")
OUTLIER_QUANTILE = 0.995
WINDOW_SIZE = 30
SPLIT_RATIO = 0.8

# ------------- MODEL HYPERPARAMETERS -------------
PARAM_GRID = {
    'model_type': ['LSTM'],
    'units': [50, 100, 150],
    'dropout': [0.2, 0.3, 0.4],
    'batch_size': [32, 64],
    'epochs': [100],
    'patience': [15]
}

# ------------- MODEL DEFAULTS -------------
DEFAULT_LSTM_UNITS = [50, 50]
DEFAULT_UNITS = 50
DEFAULT_DROPOUT = 0.2
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10

# ------------- MONGODB -------------
MONGO_DB_ENV_KEY = "MONGO_DB_URL"
DB_NAME = "electricity_forecasting"
COLLECTION_NAME = "consumption_data"

# ------------- API CONSTANTS -------------
API_TITLE = "Electricity Consumption Forecasting API"
API_DESCRIPTION = "API for forecasting electricity consumption for Indian states using LSTM models"
API_VERSION = "1.0.0"
