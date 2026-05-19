import os
import yaml
from pathlib import Path
from src.utils.logger import logger

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Loads a configuration dictionary from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Configuration file not found at {path}. Using hardcoded default configuration.")
        return get_default_config()
        
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {path}")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration from {path}: {e}. Falling back to defaults.")
        return get_default_config()

def get_default_config() -> dict:
    """Returns a hardcoded fallback configuration dict."""
    return {
        "project": {
            "name": "Smart Traffic Management System",
            "version": "1.0.0"
        },
        "database": {
            "db_path": "outputs/traffic_management.db"
        },
        "paths": {
            "raw_data_dir": "datasets/raw",
            "processed_data_dir": "datasets/processed",
            "models_dir": "models/trained",
            "checkpoints_dir": "models/checkpoints",
            "logs_dir": "outputs/logs",
            "predictions_dir": "outputs/predictions"
        },
        "density_analysis": {
            "vehicle_weights": {
                "car": 1.0,
                "bike": 0.2,
                "bus": 2.5,
                "truck": 3.0
            },
            "thresholds": {
                "low": 30.0,
                "normal": 75.0,
                "high": 120.0
            }
        },
        "signal_optimization": {
            "min_green_time": 15,
            "max_green_time": 90,
            "base_yellow_time": 4,
            "base_red_time": 2,
            "pcu_per_second": 0.15
        },
        "ml_models": {
            "random_state": 42,
            "test_size": 0.2,
            "classification": {
                "target": "Traffic Situation",
                "features": ["CarCount", "BikeCount", "BusCount", "TruckCount", "Hour", "DayOfWeek", "IsWeekend", "IsPeakHour", "DensityScore"]
            },
            "regression": {
                "target": "Total",
                "features": ["Hour", "DayOfWeek", "IsWeekend", "IsPeakHour", "CarCount", "BikeCount", "BusCount", "TruckCount", "DensityScore"]
            }
        }
    }

# Load a global configuration object
config = load_config()
