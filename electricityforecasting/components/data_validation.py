import sys
import pandas as pd
from typing import Dict, Tuple, List

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.entity import DataValidationConfig, DataValidationArtifact
from electricityforecasting.utils.common import create_dirs, save_parquet, save_json
from electricityforecasting.utils.schema_utils import SchemaManager

class DataValidation:
    def __init__(self, config: DataValidationConfig, schema_file_path: str = "schemas/electricity_schema.yaml"):
        try:
            self.config = config
            create_dirs([self.config.root_dir])
            
            # Load schema configuration
            self.schema_manager = SchemaManager(schema_file_path)
            self.schema = self.schema_manager.schema
            
            logging.info("DataValidation initialized with schema configuration")
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error initializing DataValidation: {e}", sys) from e

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """Read data from parquet file"""
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            raise ElectricityForecastingException(f"Error reading data from {file_path}: {e}", sys) from e

    def validate_basic_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate basic schema requirements"""
        try:
            validation_errors = []
            data_quality_config = self.schema_manager.get_data_quality_config()
            
            # Check if date column exists
            date_column = self.schema_manager.get_date_column()
            if date_column not in df.columns:
                validation_errors.append(f"Date column '{date_column}' not found")
            
            # Check minimum number of records
            min_records = data_quality_config.get('expected_min_records', 365)
            if len(df) < min_records:
                validation_errors.append(f"Dataset has {len(df)} records, minimum required: {min_records}")
            
            # Check for state columns (numerical columns)
            exclude_states = self.schema_manager.get_exclude_states()
            state_columns = [col for col in df.columns if col != date_column and col not in exclude_states]
            
            if len(state_columns) == 0:
                validation_errors.append("No valid state columns found in dataset")
            
            # Check if state columns are numerical
            for col in state_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    validation_errors.append(f"State column '{col}' is not numerical")
            
            # Check for completely empty columns
            empty_columns = df.columns[df.isnull().all()].tolist()
            if empty_columns:
                validation_errors.append(f"Completely empty columns found: {empty_columns}")
            
            validation_status = len(validation_errors) == 0
            logging.info(f"Basic schema validation: {'PASSED' if validation_status else 'FAILED'}")
            
            if validation_errors:
                for error in validation_errors:
                    logging.warning(f"Schema validation error: {error}")
            
            return validation_status, validation_errors
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in schema validation: {e}", sys) from e

    def validate_date_column(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate date column specifically"""
        try:
            validation_errors = []
            date_col = self.schema_manager.get_date_column()
            date_config = self.schema_manager.get_date_validation_config()
            
            if date_col not in df.columns:
                return False, [f"Date column '{date_col}' not found"]
            
            # Try to convert to datetime
            try:
                date_series = pd.to_datetime(df[date_col], errors='coerce')
                invalid_dates = date_series.isnull().sum()
                
                if invalid_dates > 0:
                    validation_errors.append(f"Found {invalid_dates} invalid dates in '{date_col}' column")
                
                # Check for date range reasonableness
                if not date_series.empty:
                    min_date = date_series.min()
                    max_date = date_series.max()
                    
                    min_year = date_config.get('min_year', 2012)
                    if min_date.year < min_year:
                        validation_errors.append(f"Date range starts too early: {min_date} (minimum year: {min_year})")
                    
                    allow_future = date_config.get('allow_future_dates', False)
                    if not allow_future and max_date > pd.Timestamp.now():
                        validation_errors.append(f"Future dates found: {max_date}")
                
            except Exception as e:
                validation_errors.append(f"Cannot convert '{date_col}' to datetime: {e}")
            
            validation_status = len(validation_errors) == 0
            logging.info(f"Date validation: {'PASSED' if validation_status else 'FAILED'}")
            
            if validation_errors:
                for error in validation_errors:
                    logging.warning(f"Date validation error: {error}")
            
            return validation_status, validation_errors
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in date validation: {e}", sys) from e

    def validate_electricity_data_quality(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate electricity-specific data quality"""
        try:
            validation_errors = []
            data_quality_config = self.schema_manager.get_data_quality_config()
            validation_rules = self.schema_manager.get_validation_rules()
            
            # Get state columns
            date_column = self.schema_manager.get_date_column()
            exclude_states = self.schema_manager.get_exclude_states()
            state_columns = [col for col in df.columns 
                           if col != date_column and col not in exclude_states]
            
            max_missing_pct = data_quality_config.get('max_missing_percentage', 50)
            outlier_multiplier = data_quality_config.get('outlier_multiplier', 10)
            negative_allowed = validation_rules.get('negative_values_allowed', False)
            
            for state in state_columns:
                # Check for negative values
                if not negative_allowed:
                    negative_count = (df[state] < 0).sum()
                    if negative_count > 0:
                        validation_errors.append(f"State '{state}' has {negative_count} negative consumption values")
                
                # Check for extremely high values (outlier detection)
                if not df[state].empty:
                    non_null_data = df[state].dropna()
                    if len(non_null_data) > 0:
                        q99 = non_null_data.quantile(0.99)
                        q01 = non_null_data.quantile(0.01)
                        
                        # Flag suspicious outliers
                        if q99 > outlier_multiplier * q01 and q01 > 0:
                            validation_errors.append(
                                f"State '{state}' has suspicious outliers "
                                f"(99th percentile: {q99:.2f}, 1st percentile: {q01:.2f})"
                            )
                
                # Check missing value percentage
                missing_percentage = (df[state].isnull().sum() / len(df)) * 100
                if missing_percentage > max_missing_pct:
                    validation_errors.append(
                        f"State '{state}' has {missing_percentage:.1f}% missing values (>{max_missing_pct}%)"
                    )
            
            validation_status = len(validation_errors) == 0
            logging.info(f"Data quality validation: {'PASSED' if validation_status else 'FAILED'}")
            
            if validation_errors:
                for error in validation_errors:
                    logging.warning(f"Data quality error: {error}")
            
            return validation_status, validation_errors
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in data quality validation: {e}", sys) from e

    def validate_temporal_consistency(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check for unusual patterns in time series data """
        try:
            validation_errors = []
            
            # Get state columns
            date_column = self.schema_manager.get_date_column()
            exclude_states = self.schema_manager.get_exclude_states()
            state_columns = [col for col in df.columns 
                           if col != date_column and col not in exclude_states]
            
            # Check for reasonable year-over-year growth
            if len(df) >= 365:  # At least one year of data
                for state in state_columns:
                    try:
                        # Compare first and last year averages
                        first_year_avg = df[state].head(365).mean()
                        last_year_avg = df[state].tail(365).mean()
                        
                        if first_year_avg > 0 and not pd.isna(first_year_avg) and not pd.isna(last_year_avg):
                            growth_rate = (last_year_avg - first_year_avg) / first_year_avg
                            
                            # Flag if growth is too extreme (>200% or <-50%)
                            if growth_rate > 2.0:
                                validation_errors.append(
                                    f"State '{state}' shows extreme growth: {growth_rate*100:.1f}%"
                                )
                            elif growth_rate < -0.5:
                                validation_errors.append(
                                    f"State '{state}' shows extreme decline: {growth_rate*100:.1f}%"
                                )
                    except Exception:
                        # Skip states with calculation issues
                        continue
            
            validation_status = len(validation_errors) == 0
            logging.info(f"Temporal consistency check: {'PASSED' if validation_status else 'FAILED'}")
            
            if validation_errors:
                for error in validation_errors:
                    logging.warning(f"Temporal consistency warning: {error}")
            
            return validation_status, validation_errors
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in temporal consistency check: {e}", sys) from e

    def _to_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert date column to datetime and set as index"""
        try:
            date_column = self.schema_manager.get_date_column()
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            
            # Remove rows with invalid dates
            before_count = len(df)
            df = df.dropna(subset=[date_column])
            after_count = len(df)
            
            if before_count != after_count:
                logging.warning(f"Removed {before_count - after_count} rows with invalid dates")
            
            return df.set_index(date_column).sort_index()
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error converting dates: {e}", sys) from e

    def seasonal_impute(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, Dict, List]:
        """Seasonal imputation with logging based on schema configuration"""
        try:
            imputation_config = self.schema_manager.get_imputation_config()
            exclude_states = self.schema_manager.get_exclude_states()
            
            imputation_method = imputation_config.get('method', 'seasonal')
            seasonal_groupby = imputation_config.get('seasonal_groupby', 'month')
            
            missing_before = {}
            missing_after = {}
            processed_states = []

            for state in df.columns:
                if state in exclude_states:
                    continue
                
                missing_before[state] = int(df[state].isna().sum())
                
                # Only impute if there are missing values
                if missing_before[state] > 0:
                    if imputation_method == 'seasonal':
                        if seasonal_groupby == 'month':
                            groupby_series = df.index.month
                        elif seasonal_groupby == 'quarter':
                            groupby_series = df.index.quarter
                        elif seasonal_groupby == 'dayofweek':
                            groupby_series = df.index.dayofweek
                        else:
                            groupby_series = df.index.month  # Default to month
                        
                        df[state] = df[state].fillna(
                            df[state].groupby(groupby_series).transform("mean")
                        )
                    elif imputation_method == 'forward_fill':
                        df[state] = df[state].fillna(method='ffill')
                    elif imputation_method == 'backward_fill':
                        df[state] = df[state].fillna(method='bfill')
                    elif imputation_method == 'mean':
                        df[state] = df[state].fillna(df[state].mean())
                
                missing_after[state] = int(df[state].isna().sum())
                processed_states.append(state)

            # Save imputation log with more details
            log_data = []
            for state in processed_states:
                imputation_rate = 0
                if missing_before[state] > 0:
                    imputation_rate = round(
                        (missing_before[state] - missing_after[state]) / missing_before[state] * 100, 2
                    )
                
                log_data.append({
                    "state": state,
                    "missing_before": missing_before[state],
                    "missing_after": missing_after[state],
                    "imputation_rate_percent": imputation_rate,
                    "imputation_method": imputation_method,
                    "total_records": len(df)
                })
            
            pd.DataFrame(log_data).to_csv(self.config.imputation_log_file, index=False)
            logging.info(f"Imputation log saved to {self.config.imputation_log_file}")
            logging.info(f"Imputation method used: {imputation_method}")
            
            return df, missing_before, missing_after, processed_states
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in seasonal imputation: {e}", sys) from e

    def run(self, ingestion_artifact) -> DataValidationArtifact:
        """Run comprehensive data validation pipeline (without drift detection)"""
        try:
            logging.info("Starting comprehensive data validation...")
            # Load raw data
            df = self.read_data(ingestion_artifact.raw_data_file)
            logging.info(f"Loaded data with shape: {df.shape}")
            
            # 1. Basic Schema Validation
            schema_valid, schema_errors = self.validate_basic_schema(df)
            
            # 2. Date Column Validation
            date_valid, date_errors = self.validate_date_column(df)
            
            # 3. Data Quality Validation
            quality_valid, quality_errors = self.validate_electricity_data_quality(df)
            
            # 4. Temporal Consistency Check 
            temporal_valid, temporal_errors = self.validate_temporal_consistency(df)
            
            # Compile all validation errors
            all_errors = schema_errors + date_errors + quality_errors + temporal_errors
            overall_validation_status = schema_valid and date_valid and quality_valid
            
            # Log validation results
            if overall_validation_status:
                logging.info("All critical validations passed successfully!")
            else:
                logging.warning(f"Validation completed with {len(all_errors)} issues:")
                for error in all_errors:
                    logging.warning(f"  - {error}")
            
            # Save validation report
            validation_report = {
                "validation_summary": {
                    "overall_status": overall_validation_status,
                    "schema_validation": schema_valid,
                    "date_validation": date_valid,
                    "quality_validation": quality_valid,
                    "temporal_validation": temporal_valid
                },
                "validation_errors": all_errors,
                "data_shape": df.shape,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
            
            validation_report_path = self.config.root_dir / "validation_report.json"
            save_json(validation_report_path, validation_report)
            
            # Continue with data processing even if some validations fail (with warnings)
            # Convert dates and set index
            df = self._to_datetime(df)
            
            # Perform imputation
            df, missing_before, missing_after, processed_states = self.seasonal_impute(df)
            
            # Save clean data
            save_parquet(df, self.config.clean_data_file)
            
            # Create comprehensive artifact
            artifact = DataValidationArtifact(
                clean_data_file=self.config.clean_data_file,
                imputation_log_file=self.config.imputation_log_file,
                validation_status=overall_validation_status,
                missing_values_before=missing_before,
                missing_values_after=missing_after,
                excluded_states=list(self.schema_manager.get_exclude_states()),
                processed_states=processed_states,
                data_shape=df.shape
            )
            
            logging.info(f"Data validation completed. Status: {'SUCCESS' if overall_validation_status else 'COMPLETED WITH WARNINGS'}")
            logging.info(f"Clean data saved to: {self.config.clean_data_file}")
            logging.info(f"Final data shape: {df.shape}")
            
            return artifact
            
        except Exception as e:
            raise ElectricityForecastingException(f"Data validation failed: {e}", sys) from e
