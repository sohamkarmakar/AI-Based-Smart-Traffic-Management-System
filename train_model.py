import sys
from src.utils.logger import logger
from src.prediction.train import TrafficModelTrainer

def main():
    """Main entrypoint for model training pipeline."""
    logger.info("Initializing Model Training Script...")
    try:
        trainer = TrafficModelTrainer()
        report = trainer.execute_training()
        logger.info("Training pipeline successfully finished! Check 'outputs/reports/model_training_report.json' for full scores.")
    except Exception as e:
        logger.critical(f"Critical failure during model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
