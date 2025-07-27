from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


@dataclass
class DataIngestionArtifact:
    """Data ingestion artifacts"""
    raw_data_file: Path
    ingestion_status: bool
    total_records: int
    columns: List[str]
    data_summary: Dict[str, Any]


@dataclass
class DataValidationArtifact:
    """Data validation artifacts"""
    clean_data_file: Path
    imputation_log_file: Path
    validation_status: bool
    missing_values_before: Dict[str, int]
    missing_values_after: Dict[str, int]
    excluded_states: List[str]
    processed_states: List[str]
    data_shape: tuple


@dataclass
class DataTransformationArtifact:
    """Data transformation artifacts"""
    transformed_data_file: Path
    transformation_status: bool
    outlier_threshold: Dict[str, float]
    transformation_applied: List[str]
    final_shape: tuple


@dataclass
class ModelTrainerArtifact:
    """Model training artifacts"""
    models_directory: Path
    training_status: bool
    successful_states: List[str]
    failed_states: List[str]
    training_summary: Dict[str, Dict[str, float]]
    best_performing_state: Optional[str]
    worst_performing_state: Optional[str]
    average_mae: float
    average_rmse: float


@dataclass
class StateModelArtifact:
    """Individual state model artifacts"""
    state_name: str
    model_file: Path
    scaler_file: Path
    metrics: Dict[str, float]
    best_params: Dict[str, Any]
    training_history: Optional[Any]
    model_size: str
    training_time: float


@dataclass
class ModelEvaluationArtifact:
    """Model evaluation artifacts"""
    evaluation_status: bool
    metrics_file: Path
    plots_directory: Path
    overall_metrics: Dict[str, float]
    state_wise_metrics: Dict[str, Dict[str, float]]
    model_comparison: Dict[str, Any]


@dataclass
class PredictionArtifact:
    """Prediction artifacts"""
    state_name: str
    predictions: Dict[str, float]
    prediction_dates: List[str]
    forecast_days: int
    average_consumption: float
    confidence_score: Optional[float]
    model_used: str
    prediction_time: str
