from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion component"""
    root_dir: Path
    mongo_url: str
    database_name: str
    collection_name: str
    raw_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for data validation component"""
    root_dir: Path
    raw_data_file: Path
    clean_data_file: Path
    imputation_log_file: Path
    exclude_states: tuple
    date_column: str = "Dates"


@dataclass(frozen=True)
class DataTransformationConfig:
    """Configuration for data transformation component"""
    root_dir: Path
    clean_data_file: Path
    transformed_data_file: Path
    outlier_quantile: float
    log_transform: bool = True


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for model training component"""
    root_dir: Path
    transformed_data_file: Path
    window_size: int
    split_ratio: float
    lstm_units: List[int]
    dropout_rate: float
    batch_size: int
    epochs: int
    patience: int
    use_hyperparameter_tuning: bool
    param_grid: Dict[str, List]


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Configuration for model evaluation component"""
    root_dir: Path
    model_dir: Path
    test_data_file: Path
    metrics_file: Path
    plots_dir: Path


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for prediction pipeline"""
    model_dir: Path
    window_size: int
    default_forecast_days: int
    max_forecast_days: int
