import sys
import os
import json
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.entity import (
    ModelTrainerConfig, 
    ModelTrainerArtifact, 
    StateModelArtifact,
    ModelMetrics,
    HyperParameters,
    TrainingResults,
    DataSplits
)
from electricityforecasting.utils.common import create_dirs, save_json
from electricityforecasting.utils.schema_utils import SchemaManager
from electricityforecasting.components.prediction_evaluator import PredictionEvaluator


class ElectricityForecaster:
    def __init__(self, config: ModelTrainerConfig, schema_file_path: str = "schemas/electricity_schema.yaml"):
        try:
            self.config = config
            create_dirs([self.config.root_dir])
            
            # Load schema configuration for consistency
            self.schema_manager = SchemaManager(schema_file_path)
            
            # Initialize prediction evaluator
            self.prediction_evaluator = PredictionEvaluator(config)
            
            logging.info("ElectricityForecaster initialized with configuration and schema")
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error initializing ElectricityForecaster: {e}", sys) from e
    
    def get_state_columns(self, df: pd.DataFrame) -> list:
        """Get valid state columns using schema configuration"""
        try:
            exclude_states = self.schema_manager.get_exclude_states()
            state_columns = [col for col in df.columns if col not in exclude_states]
            
            logging.info(f"Found {len(state_columns)} state columns for training")
            return state_columns
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error getting state columns: {e}", sys) from e
    
    def train_test_split_timeseries(self, series, split_ratio=None):
        """Chronological train-test split for time series"""
        try:
            split_ratio = split_ratio or self.config.split_ratio
            split_point = int(len(series) * split_ratio)
            train_series = series.iloc[:split_point]
            test_series = series.iloc[split_point:]
            
            logging.debug(f"Split series: {len(train_series)} train, {len(test_series)} test")
            return train_series, test_series
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in time series split: {e}", sys) from e
    
    def scale_data(self, train_series, test_series):
        """Scale data using MinMaxScaler fitted on training data only"""
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            
            train_values = train_series.values.reshape(-1, 1)
            test_values = test_series.values.reshape(-1, 1)
            
            scaler.fit(train_values)
            
            train_scaled = scaler.transform(train_values).flatten()
            test_scaled = scaler.transform(test_values).flatten()
            
            logging.debug(f"Data scaled: train range [{train_scaled.min():.3f}, {train_scaled.max():.3f}]")
            return train_scaled, test_scaled, scaler
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in data scaling: {e}", sys) from e
    
    def create_windows(self, series, window_size=None):
        """Create windowed sequences for LSTM input"""
        try:
            window_size = window_size or self.config.window_size
            X, y = [], []
            data = series if isinstance(series, np.ndarray) else series.values
            
            if len(data) <= window_size:
                raise ValueError(f"Data length {len(data)} must be greater than window_size {window_size}")
            
            for i in range(len(data) - window_size):
                X.append(data[i:i + window_size])
                y.append(data[i + window_size])
            
            X = np.array(X)
            y = np.array(y)
            
            # Reshape for LSTM: (samples, timesteps, features)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            logging.debug(f"Created windows: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error creating windows: {e}", sys) from e
    
    def build_model(self, model_type='LSTM', input_shape=None, units=None, dropout=None):
        """Build LSTM model"""
        try:
            units = units or (self.config.lstm_units[0] if self.config.lstm_units else 50)
            dropout = dropout or self.config.dropout_rate
            
            model = Sequential()
            model.add(LSTM(units))  # Let Keras infer the input shape
            model.add(Dropout(dropout))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            logging.debug(f"Built {model_type} model with {units} units and {dropout} dropout")
            return model
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error building model: {e}", sys) from e
    
    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """Train LSTM model with early stopping"""
        try:
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
            
            logging.debug(f"Model training completed. Final loss: {min(history.history['loss']):.4f}")
            return model, history
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error training model: {e}", sys) from e
    
    def predict_and_evaluate_log1p(self, model, X_test, y_test, scaler, history=None, state_name=None):
        """Predict and evaluate with proper log1p inverse transform + save accuracy JSON"""
        try:
            # Get predictions
            preds = model.predict(X_test, verbose=0)
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
            rmse = root_mean_squared_error(y_true_original, preds_original)
            r2 = r2_score(y_true_original, preds_original)
            
            # Calculate MAPE, handling division by zero
            mape_values = np.abs((y_true_original - preds_original) / np.maximum(y_true_original, 1e-8))
            mape = np.mean(mape_values) * 100
            
            # Calculate accuracy percentage
            accuracy_percentage = max(0, 100 - mape)
            
            # Get training and validation losses from history if available
            training_loss = 0.0
            validation_loss = 0.0
            if history:
                training_loss = min(history.history.get('loss', [0.0]))
                validation_loss = min(history.history.get('val_loss', [0.0]))
            
            metrics = ModelMetrics(
                mae=mae, rmse=rmse, r2_score=r2, mape=mape, 
                training_loss=training_loss, validation_loss=validation_loss
            )
            
            if state_name:
                self.prediction_evaluator.save_prediction_accuracy_json(
                    predictions=preds_original,
                    actual_values=y_true_original,
                    state_name=state_name,
                    accuracy_percentage=accuracy_percentage,
                    metrics_dict={
                        'mae': mae,
                        'rmse': rmse,
                        'r2_score': r2,
                        'mape': mape,
                        'accuracy_score': accuracy_percentage
                    }
                )
            
            logging.debug(f"Evaluation metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}, Accuracy: {accuracy_percentage:.2f}%")
            return preds_original, y_true_original, metrics
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in prediction and evaluation: {e}", sys) from e
    
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
                metrics={},  
                best_params={},  
                training_history=None, 
                model_size=model_size,
                training_time=0.0  
            )
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error saving model for {state_name}: {e}", sys) from e
    
    def _hyperparameter_tuning(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Comprehensive hyperparameter tuning """
        try:
            best_val_loss = float('inf')
            best_model = None
            best_params = None
            best_history = None
            
            total_combinations = len(list(ParameterGrid(self.config.param_grid)))
            logging.info(f"Starting hyperparameter tuning with {total_combinations} combinations")
            
            for i, params in enumerate(ParameterGrid(self.config.param_grid), 1):
                try:
                    logging.debug(f"[{i}/{total_combinations}] Testing: {params}")
                    
                    model, history = self.train_model(
                        X_train, y_train, X_val, y_val, **params, verbose=0
                    )
                    
                    val_result = model.evaluate(X_val, y_val, verbose=0)
                    val_loss = val_result[0] if isinstance(val_result, list) else val_result
                    
                    logging.debug(f"Validation loss: {val_loss:.6f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model
                        best_params = HyperParameters(**params)
                        best_history = history
                        logging.debug("New best model found!")
                        
                except Exception as e:
                    logging.warning(f"Error with params {params}: {e}")
                    continue
            
            if best_model is None:
                raise ValueError("No successful hyperparameter combination found")
            
            logging.info(f"Best hyperparameters found with validation loss: {best_val_loss:.6f}")
            return best_model, best_params, best_history
            
        except Exception as e:
            raise ElectricityForecastingException(f"Error in hyperparameter tuning: {e}", sys) from e
    
    def forecast_single_state(self, df: pd.DataFrame, state_name: str) -> StateModelArtifact:
        """Complete forecasting pipeline for a single state"""
        start_time = time.time()
        
        try:
            logging.info(f"Processing state: {state_name}")
            
            # Validate state exists in data
            if state_name not in df.columns:
                raise ValueError(f"State '{state_name}' not found in data columns")
            
            # Data preparation
            state_series = df[state_name].dropna()
            
            if len(state_series) < self.config.window_size + 100:  # Minimum data requirement
                raise ValueError(f"Insufficient data for {state_name}: {len(state_series)} records")
            
            logging.info(f"Data length for {state_name}: {len(state_series)} records")
            
            if self.config.use_hyperparameter_tuning:
                # Three-way split for hyperparameter tuning
                n = len(state_series)
                n_train = int(n * 0.7)
                n_val = int(n * 0.85)
                
                train_series = state_series.iloc[:n_train]
                val_series = state_series.iloc[n_train:n_val]
                test_series = state_series.iloc[n_val:]
                
                logging.info(f"Data splits - Train: {len(train_series)}, Val: {len(val_series)}, Test: {len(test_series)}")
                
                # Scale data
                train_scaled, temp_scaled, scaler = self.scale_data(train_series, 
                                                                  pd.concat([val_series, test_series]))
                val_scaled = scaler.transform(val_series.values.reshape(-1, 1)).flatten()
                test_scaled = scaler.transform(test_series.values.reshape(-1, 1)).flatten()
                
                # Create windows
                X_train, y_train = self.create_windows(train_scaled)
                X_val, y_val = self.create_windows(val_scaled)
                X_test, y_test = self.create_windows(test_scaled)
                
                # Hyperparameter tuning
                logging.info(f"Starting hyperparameter tuning for {state_name}")
                best_model, best_params, best_history = self._hyperparameter_tuning(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
                model = best_model
                history = best_history
                
                # Create data splits info
                data_splits = DataSplits(
                    total_samples=len(state_series),
                    train_samples=len(X_train),
                    validation_samples=len(X_val),
                    test_samples=len(X_test),
                    window_size=self.config.window_size
                )
            
            # Predictions and evaluation
            logging.info(f"Evaluating model for {state_name}")
            preds, actuals, metrics = self.predict_and_evaluate_log1p(
                model, X_test, y_test, scaler, history, state_name  # ADDED state_name parameter
            )
            
            # Save model and scaler
            state_artifact = self.save_model_and_scaler(model, scaler, state_name)
            
            # Update artifact with results
            training_time = time.time() - start_time
            state_artifact.metrics = metrics.to_dict()
            state_artifact.best_params = best_params.to_dict()
            state_artifact.training_time = training_time
            
            # Save individual state training results
            training_results = TrainingResults(
                state_name=state_name,
                model_metrics=metrics,
                hyperparameters=best_params,
                training_time=training_time,
                data_splits=data_splits,
                model_file_path=str(state_artifact.model_file),
                scaler_file_path=str(state_artifact.scaler_file)
            )
            
            # Save individual state summary
            state_summary_file = self.config.root_dir / f"{state_name.replace(' ', '_')}_summary.json"
            save_json(state_summary_file, training_results.get_summary())
            
            logging.info(f"{state_name} training completed successfully!")
            logging.info(f"Metrics - MAE: {metrics.mae:.2f}, RMSE: {metrics.rmse:.2f}, R2: {metrics.r2_score:.3f}")
            logging.info(f"Training time: {training_time:.2f} seconds")
            
            return state_artifact
            
        except Exception as e:
            logging.error(f"Error processing {state_name}: {e}")
            raise ElectricityForecastingException(f"Error processing {state_name}: {e}", sys) from e
    
    def run(self, transformation_artifact) -> ModelTrainerArtifact:
        """Run model training for all states"""
        try:
            logging.info("Starting model training for all states...")
            
            # Load transformed data
            df = pd.read_parquet(transformation_artifact.transformed_data_file)
            logging.info(f"Loaded transformed data with shape: {df.shape}")
            
            # Get valid state columns using schema
            states_to_process = self.get_state_columns(df)
            
            if not states_to_process:
                raise ValueError("No valid state columns found for training")
            
            logging.info(f"Starting training for {len(states_to_process)} states: {states_to_process}")
            
            # Initialize tracking variables
            successful_states = []
            failed_states = []
            training_summary = {}
            all_mae = []
            all_rmse = []
            all_r2 = []
            total_training_time = 0
            prediction_accuracy_reports = {}  # NEW: Track accuracy reports
            
            # Train models for each state
            for i, state in enumerate(states_to_process, 1):
                logging.info(f"\n[{i}/{len(states_to_process)}] Processing {state}...")
                
                try:
                    state_artifact = self.forecast_single_state(df, state)
                    successful_states.append(state)
                    
                    # Add to summary
                    training_summary[state] = state_artifact.metrics
                    all_mae.append(state_artifact.metrics['mae'])
                    all_rmse.append(state_artifact.metrics['rmse'])
                    all_r2.append(state_artifact.metrics['r2_score'])
                    total_training_time += state_artifact.training_time
                    
                    # Track accuracy report path
                    state_folder = self.config.root_dir / state.replace(' ', '_')
                    accuracy_report_path = state_folder / f"prediction_accuracy_{state.replace(' ', '_')}.json"
                    if accuracy_report_path.exists():
                        prediction_accuracy_reports[state] = str(accuracy_report_path)
                    
                    logging.info(f"✓ {state} completed successfully")
                    
                except Exception as e:
                    logging.error(f"✗ Failed to process {state}: {e}")
                    failed_states.append(state)
                    continue
            
            # Calculate overall metrics
            average_mae = np.mean(all_mae) if all_mae else 0.0
            average_rmse = np.mean(all_rmse) if all_rmse else 0.0
            average_r2 = np.mean(all_r2) if all_r2 else 0.0
            
            # Find best and worst performing states
            best_performing_state = None
            worst_performing_state = None
            
            if training_summary:
                best_performing_state = min(training_summary.keys(), 
                                          key=lambda x: training_summary[x]['mae'])
                worst_performing_state = max(training_summary.keys(), 
                                           key=lambda x: training_summary[x]['mae'])
            
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
                average_rmse=average_rmse,
                prediction_accuracy_reports=prediction_accuracy_reports  # NEW: Include accuracy reports
            )
            
            # Create combined accuracy report for all states
            if prediction_accuracy_reports:
                all_state_accuracy_data = {}
                for state, report_path in prediction_accuracy_reports.items():
                    try:
                        with open(report_path, 'r') as f:
                            state_data = json.load(f)
                            all_state_accuracy_data[state] = state_data['accuracy_metrics']
                    except Exception as e:
                        logging.warning(f"Could not load accuracy data for {state}: {e}")
                
                # Create combined report
                self.prediction_evaluator.create_combined_accuracy_report(all_state_accuracy_data)
            
            # Save comprehensive training summary
            comprehensive_summary = {
                "training_overview": {
                    "total_states_attempted": len(states_to_process),
                    "successful_states": len(successful_states),
                    "failed_states": len(failed_states),
                    "success_rate": len(successful_states) / len(states_to_process) * 100,
                    "total_training_time": total_training_time
                },
                "overall_metrics": {
                    "average_mae": average_mae,
                    "average_rmse": average_rmse,
                    "average_r2": average_r2,
                    "best_performing_state": best_performing_state,
                    "worst_performing_state": worst_performing_state
                },
                "successful_states": successful_states,
                "failed_states": failed_states,
                "individual_state_metrics": training_summary,
                "prediction_accuracy_reports": prediction_accuracy_reports,  # NEW: Include in summary
                "configuration": {
                    "window_size": self.config.window_size,
                    "split_ratio": self.config.split_ratio,
                    "use_hyperparameter_tuning": self.config.use_hyperparameter_tuning,
                    "epochs": self.config.epochs,
                    "batch_size": self.config.batch_size
                }
            }
            
            summary_file = self.config.root_dir / "comprehensive_training_summary.json"
            save_json(summary_file, comprehensive_summary)
            
            # Log final results
            logging.info(f"\n{'='*60}")
            logging.info(f"TRAINING PIPELINE COMPLETED")
            logging.info(f"{'='*60}")
            logging.info(f"Total states attempted: {len(states_to_process)}")
            logging.info(f"Successful: {len(successful_states)}")
            logging.info(f"Failed: {len(failed_states)}")
            logging.info(f"Success rate: {len(successful_states) / len(states_to_process) * 100:.1f}%")
            logging.info(f"Average MAE: {average_mae:.2f}")
            logging.info(f"Average RMSE: {average_rmse:.2f}")
            logging.info(f"Average R²: {average_r2:.3f}")
            logging.info(f"Total training time: {total_training_time:.2f} seconds")
            
            if best_performing_state:
                logging.info(f"Best performing state: {best_performing_state} (MAE: {training_summary[best_performing_state]['mae']:.2f})")
            
            if worst_performing_state:
                logging.info(f"Worst performing state: {worst_performing_state} (MAE: {training_summary[worst_performing_state]['mae']:.2f})")
            
            if failed_states:
                logging.warning(f"Failed states: {failed_states}")
            
            logging.info(f"Models and summaries saved to: {self.config.root_dir}")
            logging.info(f"Prediction accuracy reports generated for each state")
            
            return artifact
            
        except Exception as e:
            logging.error(f"Error in model training pipeline: {e}")
            raise ElectricityForecastingException(f"Model training pipeline failed: {e}", sys) from e
