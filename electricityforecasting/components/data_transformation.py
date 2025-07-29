import sys
import numpy as np
import pandas as pd
from pathlib import Path

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.entity import DataTransformationConfig, DataTransformationArtifact
from electricityforecasting.utils.common import create_dirs, save_parquet
from electricityforecasting.utils.schema_utils import SchemaManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, schema_file_path: str = "schemas/electricity_schema.yaml"):
        try:
            self.config = config
            create_dirs([self.config.root_dir])
            
            # Load schema configuration for consistency
            self.schema_manager = SchemaManager(schema_file_path)
            
            logging.info("DataTransformation initialized with schema configuration")
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error initializing DataTransformation: {e}", sys) from e

    def _get_state_columns(self, df: pd.DataFrame) -> list:
        """Get valid state columns (exclude excluded states)"""
        try:
            exclude_states = self.schema_manager.get_exclude_states()
            
            # All columns are state columns since datetime is now the index
            state_columns = [col for col in df.columns if col not in exclude_states]
            
            logging.info(f"Found {len(state_columns)} state columns for transformation")
            return state_columns
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error getting state columns: {e}", sys) from e

    def _outlier_log_transform(self, df: pd.DataFrame) -> tuple:
        """Apply outlier capping and log1p transformation"""
        try:
            
            state_columns = self._get_state_columns(df)
            
            
            state_data = df[state_columns].copy()
            
            
            df_transformed = df.copy()
            
            # Cap each state independently using percentiles
            outlier_threshold = self.config.outlier_quantile  
            lower_threshold = 1 - outlier_threshold 
            
            outlier_thresholds = {}  
            
            for state in state_columns:
                lower_bound = state_data[state].quantile(lower_threshold)
                upper_bound = state_data[state].quantile(outlier_threshold)
                
                # Store thresholds for artifact
                outlier_thresholds[state] = {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                # Apply capping
                df_transformed[state] = state_data[state].clip(
                    lower=lower_bound, 
                    upper=upper_bound
                )
                
                #Log the capping statistics
                capped_low = (state_data[state] < lower_bound).sum()
                capped_high = (state_data[state] > upper_bound).sum()
                total_capped = capped_low + capped_high
                
                if total_capped > 0:
                    logging.info(f"Capped {total_capped} outliers in {state} (Low: {capped_low}, High: {capped_high})")
            
            transformations_applied = ["outlier_capping_bilateral"]
            
            # Apply log1p transformation if enabled
            if self.config.log_transform:
                if (df_transformed[state_columns] < 0).any().any():
                    logging.warning("Found negative values before log transform - handling gracefully")
                    # Add small constant to ensure positive values
                    df_transformed[state_columns] = df_transformed[state_columns].abs() + 1
                
                df_transformed[state_columns] = np.log1p(df_transformed[state_columns])
                transformations_applied.append("log1p_transform")
            
            logging.info(f"Transformations applied: {transformations_applied}")
            
            return df_transformed, outlier_thresholds, transformations_applied
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in transformation: {e}", sys) from e

    def run(self, validation_artifact) -> DataTransformationArtifact:
        """Run data transformation pipeline"""
        try:
            logging.info("Starting data transformation pipeline...")

            df = pd.read_parquet(validation_artifact.clean_data_file)
            logging.info(f"Loaded clean data with shape: {df.shape}")

            df_transformed, outlier_threshold, transformations_applied = self._outlier_log_transform(df)

            save_parquet(df_transformed, self.config.transformed_data_file)

            artifact = DataTransformationArtifact(
                transformed_data_file=self.config.transformed_data_file,
                transformation_status=True,
                outlier_threshold=outlier_threshold,
                transformation_applied=transformations_applied,
                final_shape=df_transformed.shape
            )
            
            logging.info(f"Data transformation completed successfully. Final shape: {df_transformed.shape}")
            
            return artifact
            
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise ElectricityForecastingException(f"Data transformation failed: {e}", sys) from e
