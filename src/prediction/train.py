import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor

from src.utils.logger import logger
from src.utils.config_loader import config

# Define ordinal target mapping for classification
SITUATION_MAPPING = {"low": 0, "normal": 1, "high": 2, "heavy": 3}
REVERSE_SITUATION_MAPPING = {v: k for k, v in SITUATION_MAPPING.items()}

class TrafficModelTrainer:
    """Manages training, hyperparameter tuning, evaluation, and serialization of traffic prediction models."""

    def __init__(self):
        self.processed_dir = Path(config["paths"]["processed_data_dir"])
        self.models_dir = Path(config["paths"]["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_features = config["ml_models"]["classification"]["features"]
        self.class_target = config["ml_models"]["classification"]["target"]
        self.reg_features = config["ml_models"]["regression"]["features"]
        self.reg_target = config["ml_models"]["regression"]["target"]

    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads train, test, and validation CSVs."""
        train_df = pd.read_csv(self.processed_dir / "train.csv")
        test_df = pd.read_csv(self.processed_dir / "test.csv")
        val_df = pd.read_csv(self.processed_dir / "validation.csv")
        return train_df, test_df, val_df

    def prepare_classification_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Extracts classification features and encodes target situation."""
        X = df[self.class_features]
        # Encode target label (ordinal)
        y = df[self.class_target].map(SITUATION_MAPPING).fillna(1).astype(int) # Default to 1 (normal)
        return X, y

    def prepare_regression_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Extracts regression features and target volume."""
        X = df[self.reg_features]
        y = df[self.reg_target]
        return X, y

    def train_classification(self, X_train, y_train, X_test, y_test, X_val, y_val) -> dict:
        """Trains multiple classification models and returns evaluations."""
        logger.info("Starting Classification model training...")
        
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            "xgboost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training Classification Model: {name}")
            
            # Create a pipeline with scaling
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", model)
            ])
            
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred_test = pipeline.predict(X_test)
            y_pred_val = pipeline.predict(X_val)
            
            # Evaluate
            acc_test = accuracy_score(y_test, y_pred_test)
            acc_val = accuracy_score(y_val, y_pred_val)
            
            # Classification Report
            report_test = classification_report(y_test, y_pred_test, output_dict=True)
            
            # Compute Confusion Matrix
            cm_test = confusion_matrix(y_test, y_pred_test).tolist()
            
            # Save Pipeline
            model_path = self.models_dir / f"clf_{name}.joblib"
            joblib.dump(pipeline, model_path)
            
            # Extract Feature Importance (if supported)
            feature_importance = {}
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_importance = dict(zip(self.class_features, [float(x) for x in importances]))
            elif hasattr(model, "coef_"):
                # Use absolute coef value as proxy for importance in logistic regression
                coefs = np.abs(model.coef_[0])
                normalized_coefs = coefs / np.sum(coefs)
                feature_importance = dict(zip(self.class_features, [float(x) for x in normalized_coefs]))

            results[name] = {
                "accuracy_test": float(acc_test),
                "accuracy_validation": float(acc_val),
                "precision_weighted": float(report_test["weighted avg"]["precision"]),
                "recall_weighted": float(report_test["weighted avg"]["recall"]),
                "f1_weighted": float(report_test["weighted avg"]["f1-score"]),
                "confusion_matrix_test": cm_test,
                "feature_importance": feature_importance,
                "saved_path": str(model_path)
            }
            logger.info(f"Classification Model '{name}' - Test Accuracy: {acc_test:.4f}, Val Accuracy: {acc_val:.4f}")
            
        return results

    def train_regression(self, X_train, y_train, X_test, y_test, X_val, y_val) -> dict:
        """Trains multiple regression models and returns evaluations."""
        logger.info("Starting Regression model training...")
        
        models = {
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "xgboost": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training Regression Model: {name}")
            
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", model)
            ])
            
            pipeline.fit(X_train, y_train)
            
            # Predictions
            y_pred_test = pipeline.predict(X_test)
            y_pred_val = pipeline.predict(X_val)
            
            # Evaluate
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)
            
            # Save Pipeline
            model_path = self.models_dir / f"reg_{name}.joblib"
            joblib.dump(pipeline, model_path)
            
            # Feature Importance
            feature_importance = {}
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_importance = dict(zip(self.reg_features, [float(x) for x in importances]))
            elif hasattr(model, "coef_"):
                coefs = np.abs(model.coef_)
                normalized_coefs = coefs / np.sum(coefs)
                feature_importance = dict(zip(self.reg_features, [float(x) for x in normalized_coefs]))

            results[name] = {
                "rmse_test": float(rmse_test),
                "rmse_validation": float(rmse_val),
                "mae_test": float(mae_test),
                "r2_test": float(r2_test),
                "feature_importance": feature_importance,
                "saved_path": str(model_path)
            }
            logger.info(f"Regression Model '{name}' - Test RMSE: {rmse_test:.4f}, Val RMSE: {rmse_val:.4f}, R2: {r2_test:.4f}")
            
        return results

    def execute_training(self):
        """Runs the end-to-end model training process."""
        train_df, test_df, val_df = self.load_datasets()
        
        # Classification
        X_train_c, y_train_c = self.prepare_classification_data(train_df)
        X_test_c, y_test_c = self.prepare_classification_data(test_df)
        X_val_c, y_val_c = self.prepare_classification_data(val_df)
        
        clf_results = self.train_classification(
            X_train_c, y_train_c, X_test_c, y_test_c, X_val_c, y_val_c
        )
        
        # Regression
        X_train_r, y_train_r = self.prepare_regression_data(train_df)
        X_test_r, y_test_r = self.prepare_regression_data(test_df)
        X_val_r, y_val_r = self.prepare_regression_data(val_df)
        
        reg_results = self.train_regression(
            X_train_r, y_train_r, X_test_r, y_test_r, X_val_r, y_val_r
        )
        
        # Save evaluation report
        report = {
            "classification": clf_results,
            "regression": reg_results
        }
        
        report_path = Path("outputs/reports/model_training_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Model training report saved to {report_path}")
        return report

if __name__ == "__main__":
    trainer = TrafficModelTrainer()
    trainer.execute_training()
