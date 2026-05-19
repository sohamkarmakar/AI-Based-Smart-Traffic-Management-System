import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

def generate_screenshots():
    """Generates visual charts and plots for repository showcasing."""
    print("Generating visualizations for assets/screenshots/...")
    
    # 1. Setup folders
    screenshots_dir = Path("assets/screenshots")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.facecolor"] = "#0e1117"
    plt.rcParams["axes.facecolor"] = "#1e293b"
    plt.rcParams["text.color"] = "#f3f4f6"
    plt.rcParams["axes.labelcolor"] = "#f3f4f6"
    plt.rcParams["xtick.color"] = "#9ca3af"
    plt.rcParams["ytick.color"] = "#9ca3af"
    plt.rcParams["grid.color"] = "#334155"
    
    # Load processed data
    validation_path = Path("datasets/processed/validation.csv")
    if not validation_path.exists():
        print(f"Validation dataset not found at {validation_path}. Run analytics script first.")
        return
        
    df = pd.read_csv(validation_path)
    
    # Chart 1: Traffic Density Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="DensityScore", hue="Traffic Situation", kde=True, multiple="stack", 
                 palette=["#10b981", "#3b82f6", "#f59e0b", "#ef4444"])
    plt.title("Road Occupancy (PCU Density Score) by Situation Class", fontsize=14, color="#38bdf8", pad=15)
    plt.xlabel("Passenger Car Unit (PCU) Density Score")
    plt.ylabel("Observation Count")
    plt.tight_layout()
    plt.savefig(screenshots_dir / "traffic_density_distribution.png", dpi=150, facecolor="#0e1117")
    plt.close()
    
    # Chart 2: Hourly Congestion Profile
    plt.figure(figsize=(10, 5))
    hourly_avg = df.groupby("Hour")["DensityScore"].mean().reset_index()
    plt.plot(hourly_avg["Hour"], hourly_avg["DensityScore"], color="#8b5cf6", linewidth=3, marker="o", markersize=6)
    plt.fill_between(hourly_avg["Hour"], hourly_avg["DensityScore"], color="#8b5cf6", alpha=0.15)
    plt.title("City Hourly Traffic Volume Profile", fontsize=14, color="#38bdf8", pad=15)
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Mean PCU Traffic Density")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(screenshots_dir / "hourly_congestion_profile.png", dpi=150, facecolor="#0e1117")
    plt.close()

    # Chart 3: Confusion Matrix
    report_path = Path("outputs/reports/model_training_report.json")
    if report_path.exists():
        with open(report_path, "r") as f:
            metrics = json.load(f)
            
        cm = metrics["classification"]["xgboost"]["confusion_matrix_test"]
        classes = ["Low", "Normal", "High", "Heavy"]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
                    cbar=False, annot_kws={"size": 14})
        plt.ylabel("Actual Congestion Class", fontsize=12)
        plt.xlabel("Predicted Congestion Class", fontsize=12)
        plt.title("XGBoost Confusion Matrix (Test Set)", fontsize=14, color="#38bdf8", pad=15)
        plt.tight_layout()
        plt.savefig(screenshots_dir / "confusion_matrix.png", dpi=150, facecolor="#0e1117")
        plt.close()

        # Chart 4: Feature Importance
        imp_dict = metrics["classification"]["xgboost"]["feature_importance"]
        if imp_dict:
            plt.figure(figsize=(10, 5))
            sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1])
            feats, imps = zip(*sorted_imp)
            plt.barh(feats, imps, color="#3b82f6")
            plt.title("XGBoost Feature Importance Ranking", fontsize=14, color="#38bdf8", pad=15)
            plt.xlabel("Relative Information Gain")
            plt.tight_layout()
            plt.savefig(screenshots_dir / "feature_importance.png", dpi=150, facecolor="#0e1117")
            plt.close()

    print("All visualizations created and saved to assets/screenshots/ successfully.")

if __name__ == "__main__":
    generate_screenshots()
