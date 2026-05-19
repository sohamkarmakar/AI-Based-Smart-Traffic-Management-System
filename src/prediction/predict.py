import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from src.utils.logger import logger
from src.utils.config_loader import config
from src.database.db_manager import DatabaseManager
from src.prediction.train import REVERSE_SITUATION_MAPPING

class TrafficPredictor:
    """Handles real-time model inference and database logging for auditing predictions."""

    def __init__(self, db_manager: DatabaseManager = None):
        self.models_dir = Path(config["paths"]["models_dir"])
        self.db_manager = db_manager if db_manager is not None else DatabaseManager()
        self._loaded_models = {}

    def _load_model(self, model_key: str):
        """Loads a model from disk or cache."""
        if model_key not in self._loaded_models:
            model_path = self.models_dir / f"{model_key}.joblib"
            if not model_path.exists():
                logger.error(f"Trained model not found at {model_path}. Make sure to run training first.")
                raise FileNotFoundError(f"Model file missing: {model_path}")
            
            try:
                self._loaded_models[model_key] = joblib.load(model_path)
                logger.info(f"Loaded model '{model_key}' into memory.")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {e}")
                raise e
        return self._loaded_models[model_key]

    def predict_congestion(self, features: dict, model_name: str = "xgboost") -> tuple[str, float]:
        """Predicts the traffic congestion level (classification).
        
        Args:
            features: Dictionary containing counts and time info:
                      {'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Hour', 'DayOfWeek', 'IsWeekend', 'IsPeakHour', 'DensityScore'}
            model_name: Name of classifier model (e.g. 'xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression')
            
        Returns:
            predicted_situation: str (e.g., 'low', 'normal', 'high', 'heavy')
            confidence: float (highest class probability)
        """
        model_key = f"clf_{model_name}"
        pipeline = self._load_model(model_key)
        
        # Prepare feature DataFrame matching training columns
        features_order = config["ml_models"]["classification"]["features"]
        df_input = pd.DataFrame([features])[features_order]
        
        # Run prediction
        pred_encoded = pipeline.predict(df_input)[0]
        predicted_situation = REVERSE_SITUATION_MAPPING.get(int(pred_encoded), "normal")
        
        # Extract confidence/probability
        confidence = 1.0
        try:
            probs = pipeline.predict_proba(df_input)[0]
            confidence = float(np.max(probs))
        except (AttributeError, IndexError):
            # Fallback if model doesn't support predict_proba
            pass
            
        return predicted_situation, confidence

    def predict_volume(self, features: dict, model_name: str = "xgboost") -> float:
        """Predicts total traffic volume (regression).
        
        Args:
            features: Dictionary containing:
                      {'Hour', 'DayOfWeek', 'IsWeekend', 'IsPeakHour', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'DensityScore'}
            model_name: Regressor model name ('xgboost', 'random_forest', 'ridge')
            
        Returns:
            predicted_volume: float
        """
        model_key = f"reg_{model_name}"
        pipeline = self._load_model(model_key)
        
        # Prepare feature DataFrame matching training columns
        features_order = config["ml_models"]["regression"]["features"]
        df_input = pd.DataFrame([features])[features_order]
        
        # Run prediction
        pred_volume = float(pipeline.predict(df_input)[0])
        # Ensure predicted count is non-negative
        return max(0.0, pred_volume)

    def run_inference_and_log(self, features: dict, clf_model: str = "xgboost", reg_model: str = "xgboost") -> dict:
        """Runs classification and regression inference, logs the predictions to SQLite, and returns outputs."""
        predicted_situation, confidence = self.predict_congestion(features, model_name=clf_model)
        predicted_volume = self.predict_volume(features, model_name=reg_model)
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to Database
        self.db_manager.insert_prediction(
            prediction_time=now_str,
            input_features=features,
            predicted_situation=predicted_situation,
            predicted_volume=predicted_volume,
            model_type=f"classification:{clf_model} | regression:{reg_model}",
            confidence_score=confidence
        )
        
        return {
            "prediction_time": now_str,
            "situation": predicted_situation,
            "volume": round(predicted_volume, 1),
            "confidence": round(confidence, 3)
        }
