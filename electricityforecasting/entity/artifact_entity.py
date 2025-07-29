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
    
    # New fields for comprehensive validation
    validation_errors: List[str] = None
    drift_report: Dict[str, Any] = None
    drift_status: bool = True


@dataclass
class DataTransformationArtifact:
    transformed_data_file: str
    transformation_status: bool
    outlier_threshold: dict
    transformation_applied: list
    final_shape: tuple
    accuracy_metrics: dict = None  



@dataclass
class ModelTrainerArtifact:
    models_directory: str
    training_status: bool
    successful_states: list
    failed_states: list
    training_summary: dict
    best_performing_state: str
    worst_performing_state: str
    average_mae: float
    average_rmse: float
    prediction_accuracy_reports: dict = None  



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
