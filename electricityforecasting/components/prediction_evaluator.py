import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.utils.common import create_dirs


class PredictionEvaluator:
    """Simple prediction accuracy calculation and JSON output"""
    
    def __init__(self, config):
        self.config = config
    
    def save_prediction_accuracy_json(self, predictions, actual_values, state_name, 
                                    accuracy_percentage, metrics_dict):
        """Save simple prediction results and accuracy to JSON file"""
        try:
            # Simple results data
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'state': state_name,
                'model_type': 'LSTM',
                'total_predictions': len(predictions),
                'accuracy_score': accuracy_percentage,
                'metrics': {
                    'mae': metrics_dict.get('mae', 0.0),
                    'rmse': metrics_dict.get('rmse', 0.0),
                    'r2_score': metrics_dict.get('r2_score', 0.0),
                    'mape': metrics_dict.get('mape', 0.0)
                },
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                'actual_values': actual_values.tolist() if hasattr(actual_values, 'tolist') else list(actual_values)
            }
            
            # Save to state-specific folder
            state_folder = self.config.root_dir / state_name.replace(' ', '_')
            create_dirs([state_folder])
            
            json_output_path = state_folder / f"prediction_accuracy_{state_name.replace(' ', '_')}.json"
            with open(json_output_path, 'w') as f:
                json.dump(results_data, f, indent=4, default=str)
            
            logging.info(f"Prediction accuracy saved: {json_output_path}")
            logging.info(f"Test accuracy for {state_name}: {accuracy_percentage:.2f}%")
            
            return json_output_path
            
        except Exception as e:
            logging.error(f"Error saving prediction accuracy JSON: {e}")
            return None
    
    def create_combined_accuracy_report(self, all_state_results):
        """Create simple combined accuracy report for all states"""
        try:
            if not all_state_results:
                logging.warning("No state results provided for combined report")
                return None
                
            # Calculate overall average
            avg_accuracy = np.mean([r['accuracy_score'] for r in all_state_results.values()])
            
            # Find best and worst performing states
            best_state = max(all_state_results.items(), key=lambda x: x[1]['accuracy_score'])
            worst_state = min(all_state_results.items(), key=lambda x: x[1]['accuracy_score'])
            
            combined_report = {
                'timestamp': datetime.now().isoformat(),
                'total_states': len(all_state_results),
                'average_accuracy': round(avg_accuracy, 2),
                'best_performing_state': {
                    'name': best_state[0],
                    'accuracy': best_state[1]['accuracy_score']
                },
                'worst_performing_state': {
                    'name': worst_state[0],
                    'accuracy': worst_state[1]['accuracy_score']
                },
                'all_states_results': all_state_results
            }
            
            # Save combined report
            combined_report_path = self.config.root_dir / "combined_accuracy_report.json"
            with open(combined_report_path, 'w') as f:
                json.dump(combined_report, f, indent=4, default=str)
            
            logging.info(f"Combined accuracy report saved: {combined_report_path}")
            logging.info(f"Overall average accuracy: {avg_accuracy:.2f}%")
            
            return combined_report_path
            
        except Exception as e:
            logging.error(f"Error creating combined accuracy report: {e}")
            return None
