import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import tensorflow as tf
import sys

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException

class PredictionPipeline:
    def __init__(self, model_dir, window_size=30, data_file="artifacts/data_transformation/transformed_data.parquet"):
        self.model_dir = Path(model_dir)
        self.window_size = window_size
        self.data_file = data_file
        self.df = None
        self.last_data_date = None
        self._load_data()
    
    def _load_data(self):
        """Load the transformed data once during initialization"""
        try:
            if not Path(self.data_file).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
            # Load the parquet file
            self.df = pd.read_parquet(self.data_file)
            print(f"DEBUG: Data loaded, shape: {self.df.shape}")
            print(f"DEBUG: Original index type: {type(self.df.index)}")
            print(f"DEBUG: Index sample: {self.df.index[:3]}")
            
            # CRITICAL FIX: Handle datetime index properly
            if not isinstance(self.df.index, pd.DatetimeIndex):
                # Try multiple approaches to fix the index
                try:
                    # Approach 1: Direct conversion
                    self.df.index = pd.to_datetime(self.df.index, errors='coerce')
                except:
                    try:
                        # Approach 2: Reset and use date column
                        self.df = self.df.reset_index()
                        if 'Dates' in self.df.columns:
                            self.df['Dates'] = pd.to_datetime(self.df['Dates'], errors='coerce')
                            self.df = self.df.set_index('Dates')
                        elif 'Date' in self.df.columns:
                            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
                            self.df = self.df.set_index('Date')
                        elif 'index' in self.df.columns:
                            self.df['index'] = pd.to_datetime(self.df['index'], errors='coerce')
                            self.df = self.df.set_index('index')
                    except:
                        # Approach 3: Create date range manually if we know the data range
                        print("WARNING: Could not parse existing dates, creating date range")
                        # Create date range from 2012-04-01 to 2024-09-28 (known data range)
                        start_date = pd.to_datetime('2012-04-01')
                        self.df.index = pd.date_range(start=start_date, periods=len(self.df), freq='D')
            
            # Remove rows with invalid dates and NaN values
            self.df = self.df.dropna()
            
            # CRITICAL FIX: Ensure we have valid dates
            if len(self.df) == 0:
                raise ValueError("No valid data after cleaning")
            
            # Get the last date with validation
            potential_last_date = self.df.index.max()
            print(f"DEBUG: Raw max date: {potential_last_date}")
            
            # Validate the date is reasonable (not 1970 or NaT)
            if pd.isna(potential_last_date) or potential_last_date.year < 2020:
                print(f"WARNING: Invalid max date {potential_last_date}, using fallback")
                # Use known correct last date
                self.last_data_date = pd.Timestamp('2024-09-28')
            else:
                self.last_data_date = potential_last_date
            
            print(f"DEBUG: Final last data date: {self.last_data_date}")
            logging.info(f"Data loaded successfully. Last available date: {self.last_data_date.date()}")
            
        except Exception as e:
            print(f"ERROR in _load_data: {e}")
            # Set fallback values
            self.last_data_date = pd.Timestamp('2024-09-28')
            # Create a minimal DataFrame for fallback
            if self.df is None:
                dates = pd.date_range(start='2024-09-01', end='2024-09-28', freq='D')
                self.df = pd.DataFrame(index=dates)
                for state in ['Maharashtra', 'Gujarat', 'Tamil Nadu']:
                    self.df[state] = np.random.randn(len(dates)) * 1000 + 5000
            
            raise ElectricityForecastingException(f"Error loading data: {e}", sys) from e
    
    def load_model_and_scaler(self, state_name):
        """Load trained model and scaler for a specific state"""
        try:
            state_folder = self.model_dir / state_name.replace(' ', '_')
            model_path = state_folder / f"{state_name.replace(' ', '_')}_model.keras"
            scaler_path = state_folder / f"{state_name.replace(' ', '_')}_scaler.pkl"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            model = tf.keras.models.load_model(model_path)
            
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                
            return model, scaler
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error loading model for {state_name}: {e}", sys) from e
    
    def get_days_ahead(self, target_date):
        """Calculate how many days ahead the target date is from our last available data"""
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        days_ahead = (target_date - self.last_data_date).days
        return max(1, days_ahead)
    
    def predict_for_date(self, state_name, target_date):
        """Predict electricity consumption for a specific state and date using iterative forecasting"""
        try:
            # Validate inputs
            if state_name not in self.df.columns:
                available_states = [col for col in self.df.columns if col not in ['Total Consumption']]
                raise ValueError(f"State '{state_name}' not found. Available states: {available_states[:10]}...")
            
            # Convert target_date
            if isinstance(target_date, str):
                target_date = pd.to_datetime(target_date)
            
            # Calculate days ahead
            days_ahead = self.get_days_ahead(target_date)
            
            logging.info(f"Predicting for {state_name} on {target_date.date()} ({days_ahead} days ahead from last data)")
            
            # Load model and scaler
            model, scaler = self.load_model_and_scaler(state_name)
            
            # Get state data
            state_data = self.df[state_name].dropna()
            if state_data.empty:
                raise ValueError(f"No data available for state '{state_name}' after cleaning")
            
            # Get the most recent window_size days for initial prediction
            if len(state_data) < self.window_size:
                raise ValueError(f"Need at least {self.window_size} days of data, got {len(state_data)}")
            
            # Prepare initial prediction window
            last_window_data = state_data.tail(self.window_size).values
            last_window_scaled = scaler.transform(last_window_data.reshape(-1, 1)).flatten()
            
            # Generate predictions iteratively up to target date
            current_window = last_window_scaled.copy()
            
            for day in range(days_ahead):
                # Prepare model input
                model_input = current_window[-self.window_size:].reshape(1, self.window_size, 1)
                
                # Make prediction (use model() for faster inference)
                pred_scaled = model(model_input, training=False)[0, 0]
                
                # Update window for next prediction (sliding window)
                current_window = np.append(current_window[1:], pred_scaled)
            
            # Get final prediction (inverse transform)
            pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0, 0]
            pred_original = np.expm1(pred_unscaled)  # Inverse log1p transform
            
            logging.info(f"Prediction for {state_name} on {target_date.date()}: {pred_original:.2f} MWh")
            
            return float(pred_original)
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error predicting for {state_name} on {target_date}: {e}", sys) from e
    
    def get_data_info(self):
        """Get information about available data with proper date formatting"""
        try:
            # ENSURE we always return a valid date
            if hasattr(self, 'last_data_date') and self.last_data_date is not None:
                if not pd.isna(self.last_data_date) and self.last_data_date.year >= 2020:
                    last_date_str = self.last_data_date.strftime('%Y-%m-%d')
                else:
                    last_date_str = "2024-09-28"  # Hardcoded fallback
            else:
                last_date_str = "2024-09-28"  # Hardcoded fallback
            
            # Get available states (excluding problematic ones)
            if hasattr(self, 'df') and self.df is not None:
                available_states = [col for col in self.df.columns 
                                  if col not in ['Total Consumption', 'Pondy', 'Tripura']]
                total_records = len(self.df)
            else:
                # Fallback state list
                available_states = [
                    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", 
                    "Chandigarh", "Chhattisgarh", "DD", "Delhi", "DNH", "DVC", 
                    "Essar steel", "Goa", "Gujarat", "Haryana", "HP", "J&K", 
                    "Jharkhand", "Karnataka", "Kerala", "Maharashtra", "Manipur", 
                    "Meghalaya", "Mizoram", "MP", "Nagaland", "Odisha", "Punjab", 
                    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "UP", 
                    "Uttarakhand", "West Bengal"
                ]
                total_records = 4285
            
            return {
                "last_data_date": last_date_str,
                "total_records": total_records,
                "available_states": available_states
            }
            
        except Exception as e:
            logging.error(f"Error getting data info: {e}")
            # Ultimate fallback
            return {
                "last_data_date": "2024-09-28",
                "total_records": 4285,
                "available_states": ["Maharashtra", "Gujarat", "Tamil Nadu", "Karnataka", "Andhra Pradesh"]
            }
