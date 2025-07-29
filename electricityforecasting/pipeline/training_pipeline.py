import sys
from electricityforecasting.configuration.configuration import ConfigurationManager
from electricityforecasting.components.data_ingestion import DataIngestion
from electricityforecasting.components.data_validation import DataValidation
from electricityforecasting.components.data_transformation import DataTransformation
from electricityforecasting.components.multi_state_trainer import ElectricityForecaster
from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException


class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
    
    def run_data_ingestion(self):
        """Run data ingestion stage"""
        try:
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            artifact = data_ingestion.run()
            return artifact
        except Exception as e:
            raise ElectricityForecastingException(f"Data ingestion failed: {e}", sys) from e
    
    def run_data_validation(self, ingestion_artifact):
        """Run data validation stage"""
        try:
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            artifact = data_validation.run(ingestion_artifact)
            return artifact
        except Exception as e:
            raise ElectricityForecastingException(f"Data validation failed: {e}", sys) from e
    
    def run_data_transformation(self, validation_artifact):
        """Run data transformation stage"""
        try:
            data_transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            artifact = data_transformation.run(validation_artifact)
            return artifact
        except Exception as e:
            raise ElectricityForecastingException(f"Data transformation failed: {e}", sys) from e
    
    def run_model_training(self, transformation_artifact):
        """Run model training stage"""
        try:
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ElectricityForecaster(config=model_trainer_config)
            artifact = model_trainer.run(transformation_artifact)
            return artifact
        except Exception as e:
            raise ElectricityForecastingException(f"Model training failed: {e}", sys) from e
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logging.info("==== TRAINING PIPELINE START ====")
            
            # Data ingestion
            ingestion_artifact = self.run_data_ingestion()
            
            # Data validation
            validation_artifact = self.run_data_validation(ingestion_artifact)
            
            # Data transformation
            transformation_artifact = self.run_data_transformation(validation_artifact)
            
            # Model training
            training_artifact = self.run_model_training(transformation_artifact)
            
            logging.info("==== TRAINING PIPELINE COMPLETED ====")
            
            return training_artifact
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise ElectricityForecastingException(f"Training pipeline failed: {e}", sys) from e
