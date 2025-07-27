import sys
import numpy as np
import pandas as pd

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.entity import DataTransformationConfig, DataTransformationArtifact
from electricityforecasting.utils.common import create_dirs, save_parquet

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        create_dirs([self.config.root_dir])

    def _outlier_log_transform(self, df: pd.DataFrame) -> tuple:
        """Apply outlier capping and log1p transformation"""
        try:
            # Calculate outlier thresholds
            outlier_threshold = df.quantile(self.config.outlier_quantile).to_dict()
            
            # Apply outlier capping
            df_capped = df.clip(upper=df.quantile(self.config.outlier_quantile), axis=1)
            
            # Apply log1p transformation if enabled
            transformations_applied = ["outlier_capping"]
            if self.config.log_transform:
                df_capped = np.log1p(df_capped)
                transformations_applied.append("log1p_transform")
            
            return df_capped, outlier_threshold, transformations_applied
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in transformation: {e}", sys) from e
    
    def run(self, validation_artifact) -> DataTransformationArtifact:
        """Run data transformation pipeline"""
        try:
            # Load clean data
            df = pd.read_parquet(validation_artifact.clean_data_file)
            
            # Apply transformations
            df_transformed, outlier_threshold, transformations_applied = self._outlier_log_transform(df)
            
            # Save transformed data
            save_parquet(df_transformed, self.config.transformed_data_file)
            
            # Create artifact
            artifact = DataTransformationArtifact(
                transformed_data_file=self.config.transformed_data_file,
                transformation_status=True,
                outlier_threshold=outlier_threshold,
                transformation_applied=transformations_applied,
                final_shape=df_transformed.shape
            )
            
            logging.info(f"Data transformation completed successfully. Artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise ElectricityForecastingException(f"Data transformation failed: {e}", sys) from e
