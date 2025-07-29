import sys
import yaml
from pathlib import Path
from typing import Dict, Any

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException

class SchemaManager:
    """Manages schema configuration for electricity data validation"""
    
    def __init__(self, schema_file_path: str = "schemas/electricity_schema.yaml"):
        try:
            self.schema_file_path = Path(schema_file_path)
            self.schema = self._load_schema()
            logging.info(f"Schema loaded successfully from {schema_file_path}")
        except Exception as e:
            raise ElectricityForecastingException(f"Error loading schema: {e}", sys) from e
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema from YAML file"""
        try:
            if not self.schema_file_path.exists():
                raise FileNotFoundError(f"Schema file not found: {self.schema_file_path}")
            
            with open(self.schema_file_path, 'r') as file:
                schema_data = yaml.safe_load(file)
            
            if 'electricity_data_schema' not in schema_data:
                raise ValueError("Invalid schema format: 'electricity_data_schema' key not found")
            
            return schema_data['electricity_data_schema']
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error loading schema file: {e}", sys) from e
    
    def get_date_column(self) -> str:
        """Get the date column name"""
        return self.schema.get('date_column', 'Dates')
    
    def get_required_columns(self) -> list:
        """Get list of required columns"""
        return self.schema.get('required_columns', [])
    
    def get_exclude_states(self) -> list:
        """Get list of states to exclude"""
        return self.schema.get('exclude_states', [])
    
    def get_data_quality_config(self) -> Dict[str, Any]:
        """Get data quality configuration"""
        return self.schema.get('data_quality', {})
    
    def get_date_validation_config(self) -> Dict[str, Any]:
        """Get date validation configuration"""
        return self.schema.get('date_validation', {})
    
    def get_imputation_config(self) -> Dict[str, Any]:
        """Get imputation configuration"""
        return self.schema.get('imputation', {})
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules"""
        return self.schema.get('validation_rules', {})
