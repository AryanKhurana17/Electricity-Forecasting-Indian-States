from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mae: float
    rmse: float
    r2_score: float
    mape: float
    training_loss: float
    validation_loss: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'r2_score': self.r2_score,
            'mape': self.mape,
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss
        }


@dataclass
class HyperParameters:
    """Model hyperparameters"""
    model_type: str
    units: int
    dropout: float
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'units': self.units,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'learning_rate': self.learning_rate
        }


@dataclass
class TrainingResults:
    """Complete training results for a state"""
    state_name: str
    model_metrics: ModelMetrics
    hyperparameters: HyperParameters
    training_time: float
    data_splits: Dict[str, int]
    model_file_path: str
    scaler_file_path: str
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'state': self.state_name,
            'metrics': self.model_metrics.to_dict(),
            'hyperparameters': self.hyperparameters.to_dict(),
            'training_time': self.training_time,
            'data_splits': self.data_splits
        }


@dataclass
class PredictionInput:
    """Input for prediction pipeline"""
    state_name: str
    start_date: str
    days_ahead: int
    historical_data: Optional[List[float]] = None
    use_latest_data: bool = True
    
    def validate(self) -> bool:
        """Validate prediction input"""
        if not self.state_name or not self.state_name.strip():
            return False
        if self.days_ahead < 1 or self.days_ahead > 365:
            return False
        if self.historical_data and len(self.historical_data) < 30:
            return False
        return True


@dataclass
class PredictionOutput:
    """Output from prediction pipeline"""
    state_name: str
    predictions: Dict[str, float]
    metadata: Dict[str, Any]
    status: str
    error_message: Optional[str] = None
    
    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format"""
        return {
            'status': self.status,
            'state': self.state_name,
            'predictions': self.predictions,
            'metadata': self.metadata,
            'error': self.error_message
        }
