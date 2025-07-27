import sys
import pandas as pd

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.entity import DataValidationConfig, DataValidationArtifact
from electricityforecasting.utils.common import create_dirs, save_parquet

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        create_dirs([self.config.root_dir])

    def _to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date column to datetime and set as index"""
        try:
            df[self.config.date_column] = pd.to_datetime(df[self.config.date_column], errors="coerce")
            return df.set_index(self.config.date_column).sort_index()
        except Exception as e:
            raise ElectricityForecastingException(f"Error converting dates: {e}", sys) from e

    def seasonal_impute(self, df: pd.DataFrame) -> tuple:
        """Seasonal imputation with logging"""
        try:
            months = df.index.month
            missing_before = {}
            missing_after = {}
            processed_states = []

            for state in df.columns:
                if state in self.config.exclude_states:
                    continue
                
                missing_before[state] = int(df[state].isna().sum())
                df[state] = df[state].fillna(
                    df[state].groupby(months).transform("mean")
                )
                missing_after[state] = int(df[state].isna().sum())
                processed_states.append(state)

            # Save imputation log
            log_data = []
            for state in processed_states:
                log_data.append({
                    "state": state,
                    "missing_before": missing_before[state],
                    "missing_after": missing_after[state]
                })
            
            pd.DataFrame(log_data).to_csv(self.config.imputation_log_file, index=False)
            logging.info(f"Imputation log saved to {self.config.imputation_log_file}")
            
            return df, missing_before, missing_after, processed_states
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in seasonal imputation: {e}", sys) from e

    def run(self, ingestion_artifact) -> DataValidationArtifact:
        """Run data validation pipeline"""
        try:
            # Load raw data
            df = pd.read_parquet(ingestion_artifact.raw_data_file)
            
            # Convert dates and set index
            df = self._to_datetime(df)
            
            # Perform imputation
            df, missing_before, missing_after, processed_states = self.seasonal_impute(df)
            
            # Save clean data
            save_parquet(df, self.config.clean_data_file)
            
            # Create artifact
            artifact = DataValidationArtifact(
                clean_data_file=self.config.clean_data_file,
                imputation_log_file=self.config.imputation_log_file,
                validation_status=True,
                missing_values_before=missing_before,
                missing_values_after=missing_after,
                excluded_states=list(self.config.exclude_states),
                processed_states=processed_states,
                data_shape=df.shape
            )
            
            logging.info(f"Data validation completed successfully. Artifact: {artifact}")
            return artifact
            
        except Exception as e:
            logging.error(f"Error in data validation: {e}")
            raise ElectricityForecastingException(f"Data validation failed: {e}", sys) from e
