from electricityforecasting.pipeline.training_pipeline import TrainingPipeline
from electricityforecasting.logger.logger import logging

def main():
    """Main function to run the complete training pipeline"""
    try:
        logging.info("Starting Electricity Forecasting Training Pipeline")
        
        # Initialize and run training pipeline
        pipeline = TrainingPipeline()
        training_artifact = pipeline.run_training_pipeline()
        
        # Print summary
        print(f"\n=== TRAINING COMPLETED ===")
        print(f"Successful models: {len(training_artifact.successful_states)}")
        print(f"Failed models: {len(training_artifact.failed_states)}")
        print(f"Average MAE: {training_artifact.average_mae:.4f}")
        print(f"Average RMSE: {training_artifact.average_rmse:.4f}")
        
        if training_artifact.successful_states:
            print(f"Best performing state: {training_artifact.best_performing_state}")
            print(f"Worst performing state: {training_artifact.worst_performing_state}")
        
        logging.info("Training pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()
