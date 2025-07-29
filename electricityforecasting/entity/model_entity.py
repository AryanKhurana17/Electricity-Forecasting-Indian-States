from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

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
    
    def is_better_than(self, other: 'ModelMetrics', primary_metric: str = 'mae') -> bool:
        """Compare two metrics to determine which is better"""
        if primary_metric in ['mae', 'rmse', 'mape', 'training_loss', 'validation_loss']:
            return getattr(self, primary_metric) < getattr(other, primary_metric)
        elif primary_metric == 'r2_score':
            return self.r2_score > other.r2_score
        else:
            return self.mae < other.mae

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
    
    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'HyperParameters':
        """Create HyperParameters from dictionary"""
        return cls(**{k: v for k, v in params_dict.items() if k in cls.__annotations__})

@dataclass
class DataSplits:
    """Information about data splits used in training"""
    total_samples: int
    train_samples: int
    validation_samples: int = 0
    test_samples: int = 0
    window_size: int = 30
    
    def to_dict(self) -> Dict[str, int]:
        return {
            'total_samples': self.total_samples,
            'train_samples': self.train_samples,
            'validation_samples': self.validation_samples,
            'test_samples': self.test_samples,
            'window_size': self.window_size
        }

@dataclass
class TrainingResults:
    """Complete training results for a state"""
    state_name: str
    model_metrics: ModelMetrics
    hyperparameters: HyperParameters
    training_time: float
    data_splits: DataSplits
    model_file_path: str
    scaler_file_path: str
    training_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'state': self.state_name,
            'metrics': self.model_metrics.to_dict(),
            'hyperparameters': self.hyperparameters.to_dict(),
            'training_time': self.training_time,
            'data_splits': self.data_splits.to_dict(),
            'training_timestamp': self.training_timestamp
        }

@dataclass
class PredictionInput:
    """Input for prediction pipeline"""
    state_name: str
    start_date: str
    days_ahead: int
    historical_data: Optional[List[float]] = None
    use_latest_data: bool = True
    confidence_level: float = 0.95
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate prediction input"""
        if not self.state_name or not self.state_name.strip():
            return False, "State name cannot be empty"
        if self.days_ahead < 1 or self.days_ahead > 365:
            return False, "Days ahead must be between 1 and 365"
        if self.historical_data and len(self.historical_data) < 30:
            return False, "Historical data must contain at least 30 data points"
        if not (0.5 <= self.confidence_level <= 0.99):
            return False, "Confidence level must be between 0.5 and 0.99"
        
        try:
            from datetime import datetime
            datetime.strptime(self.start_date, '%Y-%m-%d')
        except ValueError:
            return False, "Start date must be in YYYY-MM-DD format"
        
        return True, None

@dataclass
class PredictionOutput:
    """Output from prediction pipeline"""
    state_name: str
    predictions: Dict[str, float]
    metadata: Dict[str, Any]
    status: str
    error_message: Optional[str] = None
    confidence_intervals: Optional[Dict[str, Dict[str, float]]] = None
    prediction_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_api_response(self) -> Dict[str, Any]:
        """Convert to API response format"""
        response = {
            'status': self.status,
            'state': self.state_name,
            'predictions': self.predictions,
            'metadata': self.metadata,
            'prediction_timestamp': self.prediction_timestamp
        }
        
        if self.error_message:
            response['error'] = self.error_message
        
        if self.confidence_intervals:
            response['confidence_intervals'] = self.confidence_intervals
            
        return response
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary statistics of predictions"""
        if not self.predictions:
            return {}
        
        values = list(self.predictions.values())
        return {
            'min_prediction': min(values),
            'max_prediction': max(values),
            'mean_prediction': sum(values) / len(values),
            'total_predictions': len(values)
        }

# Add a new entity for model comparison
@dataclass
class ModelComparison:
    """Compare multiple models for the same state"""
    state_name: str
    models: List[TrainingResults]
    best_model_index: int = 0
    comparison_metric: str = 'mae'
    
    def __post_init__(self):
        """Find the best model after initialization"""
        if self.models:
            best_idx = 0
            best_value = getattr(self.models[0].model_metrics, self.comparison_metric)
            
            for i, model in enumerate(self.models[1:], 1):
                current_value = getattr(model.model_metrics, self.comparison_metric)
                
                # For metrics where lower is better
                if self.comparison_metric in ['mae', 'rmse', 'mape', 'training_loss', 'validation_loss']:
                    if current_value < best_value:
                        best_value = current_value
                        best_idx = i
                # For metrics where higher is better
                elif self.comparison_metric == 'r2_score':
                    if current_value > best_value:
                        best_value = current_value
                        best_idx = i
            
            self.best_model_index = best_idx
    
    def get_best_model(self) -> TrainingResults:
        """Get the best performing model"""
        return self.models[self.best_model_index]
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get comparison summary"""
        return {
            'state_name': self.state_name,
            'total_models': len(self.models),
            'best_model_index': self.best_model_index,
            'comparison_metric': self.comparison_metric,
            'best_model_summary': self.get_best_model().get_summary()
        }
