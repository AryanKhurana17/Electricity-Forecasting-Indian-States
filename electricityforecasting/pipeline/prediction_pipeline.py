import sys, os, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from pathlib import Path

from electricityforecasting.constants.constants import FORECASTING_RESULTS_DIR, WINDOW_SIZE
from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException

class StatePredictor:
    def __init__(self, base_path=None):
        self.base_path = base_path or FORECASTING_RESULTS_DIR
        self.available_states = self._get_available_states()
        logging.info(f"StatePredictor initialized with {len(self.available_states)} available states")
    
    def _get_available_states(self):
        """Get list of states that have trained models"""
        try:
            states = []
            if not self.base_path.exists():
                return states
            
            for state_folder in os.listdir(self.base_path):
                state_path = self.base_path / state_folder
                if state_path.is_dir():
                    model_file = state_path / f"{state_folder}_model.keras"
                    scaler_file = state_path / f"{state_folder}_scaler.pkl"
                    if model_file.exists() and scaler_file.exists():
                        # Convert folder name back to state name
                        state_name = state_folder.replace('_', ' ')
                        states.append(state_name)
            return sorted(states)
        except Exception as e:
            raise ElectricityForecastingException(f"Error getting available states: {e}", sys)
    
    def load_model_and_scaler(self, state_name):
        """Load saved model and scaler for a specific state - matches your exact logic"""
        try:
            state_folder = self.base_path / state_name.replace(' ', '_')
            
            # Load model
            model_path = state_folder / f'{state_name.replace(" ", "_")}_model.keras'
            model = tf.keras.models.load_model(model_path)
            
            # Load scaler
            scaler_path = state_folder / f'{state_name.replace(" ", "_")}_scaler.pkl'
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            print(f"Loaded model and scaler for {state_name}")
            return model, scaler
        except Exception as e:
            raise ElectricityForecastingException(f"Error loading model for {state_name}: {e}", sys)
    
    def predict_future(self, state_name, start_date, days_ahead=7, 
                      model=None, scaler=None, df=None, window_size=WINDOW_SIZE):
        """Your exact predict_future_safe function"""
        try:
            # Validate inputs
            if df is None:
                raise ValueError("DataFrame (df) is None. Please provide a valid DataFrame.")
            
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(df)}")
            
            if df.empty:
                raise ValueError("DataFrame is empty")
            
            if state_name not in df.columns:
                available_states = df.columns.tolist()
                raise ValueError(f"State '{state_name}' not found. Available states: {available_states}")
            
            # Convert start_date
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            
            print(f"🔮 Predicting {days_ahead} days for {state_name} from {start_date.date()}")
            
            # Load model and scaler if needed
            if model is None or scaler is None:
                model, scaler = self.load_model_and_scaler(state_name)
            
            # Safely prepare DataFrame
            df_work = df.copy()
            
            # Ensure proper datetime index
            if not isinstance(df_work.index, pd.DatetimeIndex):
                df_work.index = pd.to_datetime(df_work.index)
            df_work = df_work.dropna()
            
            # Check if state has data
            state_data = df_work[state_name].dropna()
            if state_data.empty:
                raise ValueError(f"No data available for state '{state_name}' after cleaning")
            
            # Get historical data for prediction
            historical_data = state_data[state_data.index <= start_date]
            
            if len(historical_data) < window_size:
                print(f"Using most recent {window_size} days (not enough data up to {start_date.date()})")
                historical_data = state_data.tail(window_size)
            
            if len(historical_data) < window_size:
                raise ValueError(f"Need at least {window_size} days of data, got {len(historical_data)}")
            
            # Prepare prediction window
            last_window_data = historical_data.tail(window_size).values
            last_window_scaled = scaler.transform(last_window_data.reshape(-1, 1)).flatten()
            
            # Generate predictions
            predictions_original = []
            current_window = last_window_scaled.copy()
            
            for day in range(days_ahead):
                model_input = current_window[-window_size:].reshape(1, window_size, 1)
                pred_scaled = model.predict(model_input, verbose=0)[0, 0]
                pred_unscaled = scaler.inverse_transform([[pred_scaled]])[0, 0]
                pred_original = np.expm1(pred_unscaled)
                
                predictions_original.append(pred_original)
                current_window = np.append(current_window[1:], pred_scaled)
            
            # Create results
            prediction_dates = [start_date + timedelta(days=i+1) for i in range(days_ahead)]
            predictions_dict = {
                date.strftime('%Y-%m-%d'): round(pred, 2)
                for date, pred in zip(prediction_dates, predictions_original)
            }
            
            # Print results
            print(f"\n{days_ahead}-Day Forecast for {state_name}:")
            print("-" * 50)
            for date, consumption in predictions_dict.items():
                day_name = pd.to_datetime(date).strftime('%A')
                print(f"{date} ({day_name}): {consumption} Mega Units")
            
            avg_consumption = sum(predictions_original) / len(predictions_original)
            print(f"\nAverage: {avg_consumption:.2f} Mega Units")
            print("Prediction completed!")
            
            return predictions_dict
            
        except Exception as e:
            print(f"Error: {e}")
            raise ElectricityForecastingException(f"Error in predict_future: {e}", sys)
    
    def get_available_states(self):
        """Return list of states with trained models"""
        return self.available_states
