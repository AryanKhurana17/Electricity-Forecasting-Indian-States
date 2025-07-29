import sys
from pathlib import Path
import os
from dotenv import load_dotenv

from electricityforecasting.constants.constants import *
from electricityforecasting.utils.common import read_yaml, create_dirs
from electricityforecasting.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    PredictionConfig
)
from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException

load_dotenv()

class ConfigurationManager:
    def __init__(self, config_filepath: Path = None):
        try:
            self.config = None
            if config_filepath and config_filepath.exists():
                self.config = read_yaml(config_filepath)
            
            # Create base directories
            create_dirs([ARTIFACTS_ROOT, FORECASTING_RESULTS_DIR])
            
            logging.info("ConfigurationManager initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing ConfigurationManager: {e}")
            raise ElectricityForecastingException(f"Error in ConfigurationManager initialization: {e}", sys) from e
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration"""
        try:
            create_dirs([RAW_DATA_DIR])
            
            data_ingestion_config = DataIngestionConfig(
                root_dir=RAW_DATA_DIR,
                mongo_url=os.getenv(MONGO_DB_ENV_KEY),
                database_name=DB_NAME,
                collection_name=COLLECTION_NAME,
                raw_data_file=RAW_DF_FILE
            )
            
            logging.info("Data ingestion config created successfully")
            return data_ingestion_config
            
        except Exception as e:
            logging.error(f"Error creating data ingestion config: {e}")
            raise ElectricityForecastingException(f"Error in get_data_ingestion_config: {e}", sys) from e
    
    def get_data_validation_config(self) -> DataValidationConfig:
        """Get data validation configuration"""
        try:
            create_dirs([VALIDATED_DATA_DIR])
            
            data_validation_config = DataValidationConfig(
                root_dir=VALIDATED_DATA_DIR,
                raw_data_file=RAW_DF_FILE,
                clean_data_file=CLEAN_DF_FILE,
                imputation_log_file=IMPUTATION_LOG_FILE,
                exclude_states=EXCLUDE_STATES,
                date_column="Dates"
            )
            
            logging.info("Data validation config created successfully")
            return data_validation_config
            
        except Exception as e:
            logging.error(f"Error creating data validation config: {e}")
            raise ElectricityForecastingException(f"Error in get_data_validation_config: {e}", sys) from e
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Get data transformation configuration"""
        try:
            create_dirs([TRANSFORMED_DATA_DIR])
            
            data_transformation_config = DataTransformationConfig(
                root_dir=TRANSFORMED_DATA_DIR,
                clean_data_file=CLEAN_DF_FILE,
                transformed_data_file=TRANSFORMED_DATA_DIR / "transformed_data.parquet",
                outlier_quantile=OUTLIER_QUANTILE,
                log_transform=True
            )
            
            logging.info("Data transformation config created successfully")
            return data_transformation_config
            
        except Exception as e:
            logging.error(f"Error creating data transformation config: {e}")
            raise ElectricityForecastingException(f"Error in get_data_transformation_config: {e}", sys) from e
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Get model trainer configuration"""
        try:
            create_dirs([FORECASTING_RESULTS_DIR])
            
            model_trainer_config = ModelTrainerConfig(
                root_dir=FORECASTING_RESULTS_DIR,
                transformed_data_file=TRANSFORMED_DATA_DIR / "transformed_data.parquet",
                window_size=WINDOW_SIZE,
                split_ratio=SPLIT_RATIO,
                lstm_units=DEFAULT_LSTM_UNITS,
                dropout_rate=DEFAULT_DROPOUT,
                batch_size=DEFAULT_BATCH_SIZE,
                epochs=DEFAULT_EPOCHS,
                patience=DEFAULT_PATIENCE,
                use_hyperparameter_tuning=True,
                param_grid=PARAM_GRID
            )
            
            logging.info("Model trainer config created successfully")
            return model_trainer_config
            
        except Exception as e:
            logging.error(f"Error creating model trainer config: {e}")
            raise ElectricityForecastingException(f"Error in get_model_trainer_config: {e}", sys) from e
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Get model evaluation configuration"""
        try:
            plots_dir = ARTIFACTS_ROOT / "plots"
            create_dirs([plots_dir])
            
            model_evaluation_config = ModelEvaluationConfig(
                root_dir=ARTIFACTS_ROOT / "model_evaluation",
                model_dir=FORECASTING_RESULTS_DIR,
                test_data_file=TRANSFORMED_DATA_DIR / "transformed_data.parquet",
                metrics_file=FORECASTING_RESULTS_DIR / "overall_metrics.json",
                plots_dir=plots_dir
            )
            
            logging.info("Model evaluation config created successfully")
            return model_evaluation_config
            
        except Exception as e:
            logging.error(f"Error creating model evaluation config: {e}")
            raise ElectricityForecastingException(f"Error in get_model_evaluation_config: {e}", sys) from e
    
    def get_prediction_config(self) -> PredictionConfig:
        """Get prediction configuration"""
        try:
            prediction_config = PredictionConfig(
                model_dir=FORECASTING_RESULTS_DIR,
                window_size=WINDOW_SIZE,
                default_forecast_days=7,
                max_forecast_days=365
            )
            
            logging.info("Prediction config created successfully")
            return prediction_config
            
        except Exception as e:
            logging.error(f"Error creating prediction config: {e}")
            raise ElectricityForecastingException(f"Error in get_prediction_config: {e}", sys) from e
