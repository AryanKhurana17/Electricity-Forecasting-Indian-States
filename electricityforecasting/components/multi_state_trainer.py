import sys
import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.entity import (
    ModelTrainerConfig, 
    ModelTrainerArtifact, 
    StateModelArtifact,
    ModelMetrics,
    HyperParameters,
    TrainingResults
)
from electricityforecasting.utils.common import create_dirs, save_json

class ElectricityForecaster:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        create_dirs([self.config.root_dir])
    
    def train_test_split_timeseries(self, series, split_ratio=None):
        """Chronological train-test split for time series"""
        split_ratio = split_ratio or self.config.split_ratio
        split_point = int(len(series) * split_ratio)
        train_series = series.iloc[:split_point]
        test_series = series.iloc[split_point:]
        return train_series, test_series
    
    def scale_data(self, train_series, test_series):
        """Scale data using MinMaxScaler fitted on training data only"""
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        train_values = train_series.values.reshape(-1, 1)
        test_values = test_series.values.reshape(-1, 1)
        
        scaler.fit(train_values)
        
        train_scaled = scaler.transform(train_values).flatten()
        test_scaled = scaler.transform(test_values).flatten()
        
        return train_scaled, test_scaled, scaler
    
    def create_windows(self, series, window_size=None):
        """Create windowed sequences for LSTM input"""
        window_size = window_size or self.config.window_size
        X, y = [], []
        data = series if isinstance(series, np.ndarray) else series.values
        
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM: (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y
    
    def build_model(self, model_type='LSTM', input_shape=None, units=None, dropout=None):
        """Build LSTM or GRU model"""
        window_size = self.config.window_size
        input_shape = input_shape or (window_size, 1)
        units = units or self.config.lstm_units[0] if self.config.lstm_units else 50
        dropout = dropout or self.config.dropout_rate
        
        model = Sequential()
        
        if model_type == 'LSTM':
            model.add(LSTM(units, input_shape=input_shape))
        elif model_type == 'GRU':
            model.add(GRU(units, input_shape=input_shape))
        else:
            raise ValueError("model_type must be 'LSTM' or 'GRU'")
        
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """Train LSTM/GRU model with early stopping"""
        model = self.build_model(
            model_type=kwargs.get('model_type', 'LSTM'),
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=kwargs.get('units', self.config.lstm_units[0] if self.config.lstm_units else 50),
            dropout=kwargs.get('dropout', self.config.dropout_rate)
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=kwargs.get('patience', self.config.patience),
            restore_best_weights=True,
            verbose=0
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=kwargs.get('batch_size', self.config.batch_size),
            epochs=kwargs.get('epochs', self.config.epochs),
            callbacks=[early_stopping],
            verbose=kwargs.get('verbose', 0)
        )
        
        return model, history
    
    def predict_and_evaluate_log1p(self, model, X_test, y_test, scaler):
        """Predict and evaluate with proper log1p inverse transform"""
        # Get predictions
        preds = model.predict(X_test)
        preds = preds.flatten()
        y_true = y_test.flatten()
        
        # Inverse scaling
        preds_unscaled = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        y_true_unscaled = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        
        # Inverse log1p transform
        preds_original = np.expm1(preds_unscaled)
        y_true_original = np.expm1(y_true_unscaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_original, preds_original)
        rmse = np.sqrt(mean_squared_error(y_true_original, preds_original))
        r2 = r2_score(y_true_original, preds_original)
        mape = np.mean(np.abs((y_true_original - preds_original) / y_true_original)) * 100
        
        return preds_original, y_true_original, ModelMetrics(
            mae=mae, rmse=rmse, r2_score=r2, mape=mape, 
            training_loss=0.0, validation_loss=0.0
        )
    
    def save_model_and_scaler(self, model, scaler, state_name) -> StateModelArtifact:
        """Save model and scaler for a specific state"""
        try:
            # Create state-specific folder
            state_folder = self.config.root_dir / state_name.replace(' ', '_')
            create_dirs([state_folder])
            
            # Save model (.keras format)
            model_path = state_folder / f'{state_name.replace(" ", "_")}_model.keras'
            model.save(model_path)
            
            # Save scaler
            scaler_path = state_folder / f'{state_name.replace(" ", "_")}_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # Get model size
            model_size = f"{os.path.getsize(model_path) / (1024*1024):.2f} MB"
            
            logging.info(f"Saved model and scaler for {state_name} in: {state_folder}")
            
            return StateModelArtifact(
                state_name=state_name,
                model_file=model_path,
                scaler_file=scaler_path,
                metrics={},  # Will be filled by caller
                best_params={},  # Will be filled by caller
                training_history=None,  # Will be filled by caller
                model_size=model_size,
                training_time=0.0  # Will be filled by caller
            )
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error saving model for {state_name}: {e}", sys) from e
    
    def forecast_single_state(self, df: pd.DataFrame, state_name: str) -> StateModelArtifact:
        """Complete forecasting pipeline for a single state"""
        start_time = time.time()
        
        try:
            logging.info(f"Processing: {state_name}")
            
            # Data preparation
            state_series = df[state_name]
            
            if self.config.use_hyperparameter_tuning:
                # Three-way split for hyperparameter tuning
                n = len(state_series)
                n_train = int(n * 0.7)
                n_val = int(n * 0.85)
                
                train_series = state_series.iloc[:n_train]
                val_series = state_series.iloc[n_train:n_val]
                test_series = state_series.iloc[n_val:]
                
                # Scale data
                train_scaled, temp_scaled, scaler = self.scale_data(train_series, 
                                                                  pd.concat([val_series, test_series]))
                val_scaled = scaler.transform(val_series.values.reshape(-1, 1)).flatten()
                test_scaled = scaler.transform(test_series.values.reshape(-1, 1)).flatten()
                
                # Create windows
                X_train, y_train = self.create_windows(train_scaled)
                X_val, y_val = self.create_windows(val_scaled)
                X_test, y_test = self.create_windows(test_scaled)
                
                # Hyperparameter tuning (simplified)
                best_model, best_params = self._hyperparameter_tuning(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
                model = best_model
            else:
                # Simple two-way split
                train_series, test_series = self.train_test_split_timeseries(state_series)
                
                # Scale data
                train_scaled, test_scaled, scaler = self.scale_data(train_series, test_series)
                
                # Create windows
                X_train, y_train = self.create_windows(train_scaled)
                X_test, y_test = self.create_windows(test_scaled)
                
                # Train model with default parameters
                model, history = self.train_model(X_train, y_train, X_test, y_test, verbose=1)
                best_params = HyperParameters(
                    model_type='LSTM', units=50, dropout=0.2, 
                    batch_size=32, epochs=100, patience=10
                )
            
            # Predictions and evaluation
            preds, actuals, metrics = self.predict_and_evaluate_log1p(model, X_test, y_test, scaler)
            
            # Save model and scaler
            state_artifact = self.save_model_and_scaler(model, scaler, state_name)
            
            # Update artifact with results
            training_time = time.time() - start_time
            state_artifact.metrics = metrics.to_dict()
            state_artifact.best_params = best_params.to_dict() if hasattr(best_params, 'to_dict') else best_params
            state_artifact.training_time = training_time
            
            logging.info(f"{state_name} completed successfully! MAE: {metrics.mae:.2f}, RMSE: {metrics.rmse:.2f}")
            
            return state_artifact
            
        except Exception as e:
            logging.error(f"Error processing {state_name}: {e}")
            raise ElectricityForecastingException(f"Error processing {state_name}: {e}", sys) from e
    
    def _hyperparameter_tuning(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Simplified hyperparameter tuning"""
        best_val_loss = float('inf')
        best_model = None
        best_params = None
        
        for params in ParameterGrid(self.config.param_grid):
            try:
                model, history = self.train_model(
                    X_train, y_train, X_val, y_val, **params, verbose=0
                )
                
                val_result = model.evaluate(X_val, y_val, verbose=0)
                val_loss = val_result[0] if isinstance(val_result, list) else val_result
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_params = HyperParameters(**params)
                    
            except Exception as e:
                logging.warning(f"Error with params {params}: {e}")
                continue
        
        return best_model, best_params
    
    def run(self, transformation_artifact) -> ModelTrainerArtifact:
        """Run model training for all states"""
        try:
            # Load transformed data
            df = pd.read_parquet(transformation_artifact.transformed_data_file)
            
            successful_states = []
            failed_states = []
            training_summary = {}
            all_mae = []
            all_rmse = []
            
            # Filter out excluded states and total consumption
            states_to_process = [col for col in df.columns 
                               if col not in ['Total Consumption'] + list(self.config.exclude_states if hasattr(self.config, 'exclude_states') else [])]
            
            logging.info(f"Starting training for {len(states_to_process)} states")
            
            for i, state in enumerate(states_to_process, 1):
                logging.info(f"[{i}/{len(states_to_process)}] Processing {state}...")
                
                try:
                    state_artifact = self.forecast_single_state(df, state)
                    successful_states.append(state)
                    
                    # Add to summary
                    training_summary[state] = state_artifact.metrics
                    all_mae.append(state_artifact.metrics['mae'])
                    all_rmse.append(state_artifact.metrics['rmse'])
                    
                except Exception as e:
                    logging.error(f"Failed to process {state}: {e}")
                    failed_states.append(state)
            
            # Calculate overall metrics
            average_mae = np.mean(all_mae) if all_mae else 0.0
            average_rmse = np.mean(all_rmse) if all_rmse else 0.0
            
            # Find best and worst performing states
            best_performing_state = min(training_summary.keys(), 
                                      key=lambda x: training_summary[x]['mae']) if training_summary else None
            worst_performing_state = max(training_summary.keys(), 
                                       key=lambda x: training_summary[x]['mae']) if training_summary else None
            
            # Create overall artifact
            artifact = ModelTrainerArtifact(
                models_directory=self.config.root_dir,
                training_status=len(successful_states) > 0,
                successful_states=successful_states,
                failed_states=failed_states,
                training_summary=training_summary,
                best_performing_state=best_performing_state,
                worst_performing_state=worst_performing_state,
                average_mae=average_mae,
                average_rmse=average_rmse
            )
            
            # Save training summary
            summary_file = self.config.root_dir / "training_summary.json"
            save_json(summary_file, {
                "successful_states": successful_states,
                "failed_states": failed_states,
                "training_summary": training_summary,
                "overall_metrics": {
                    "average_mae": average_mae,
                    "average_rmse": average_rmse,
                    "best_performing_state": best_performing_state,
                    "worst_performing_state": worst_performing_state
                }
            })
            
            logging.info(f"Training completed: {len(successful_states)} successful, {len(failed_states)} failed")
            return artifact
            
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise ElectricityForecastingException(f"Model training failed: {e}", sys) from e
