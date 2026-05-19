import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.logger import logger
from src.utils.config_loader import config
from src.database.db_manager import DatabaseManager

class TrafficDensityAnalyzer:
    """Computes density scores, extracts temporal features, and handles data cleaning/validation."""

    def __init__(self):
        self.weights = config["density_analysis"]["vehicle_weights"]
        self.thresholds = config["density_analysis"]["thresholds"]
        
    def calculate_density_score(self, car_count: int, bike_count: int, bus_count: int, truck_count: int) -> float:
        """Calculates a Passenger Car Unit (PCU) equivalent density score."""
        score = (
            car_count * self.weights["car"] +
            bike_count * self.weights["bike"] +
            bus_count * self.weights["bus"] +
            truck_count * self.weights["truck"]
        )
        return float(round(score, 2))

    def get_situation_from_score(self, score: float) -> str:
        """Determines traffic congestion label strictly from density score thresholds."""
        if score < self.thresholds["low"]:
            return "low"
        elif score < self.thresholds["normal"]:
            return "normal"
        elif score < self.thresholds["high"]:
            return "high"
        else:
            return "heavy"

    def parse_time_features(self, time_str: str) -> tuple[int, int, int]:
        """Parses a time string (e.g., '12:30:00 AM' or '3:15:00 PM') to extract hour, minute, and peak hour flag."""
        try:
            # Parse using pandas to be robust to varying spaces/formats
            t = pd.to_datetime(time_str.strip(), format="%I:%M:%S %p").time()
            hour = t.hour
            minute = t.minute
            
            # Peak hours: 07:00 - 09:30 and 16:30 - 19:30
            time_val = hour + (minute / 60.0)
            is_peak = 1 if ((7.0 <= time_val <= 9.5) or (16.5 <= time_val <= 19.5)) else 0
            
            return hour, minute, is_peak
        except Exception as e:
            logger.error(f"Error parsing time string '{time_str}': {e}. Using defaults.")
            return 0, 0, 0

    def encode_day_of_week(self, day_str: str) -> tuple[int, int]:
        """Encodes day name (e.g. 'Monday') to day number (0-6) and weekend flag (0/1)."""
        day_mapping = {
            "monday": (0, 0),
            "tuesday": (1, 0),
            "wednesday": (2, 0),
            "thursday": (3, 0),
            "friday": (4, 0),
            "saturday": (5, 1),
            "sunday": (6, 1)
        }
        clean_day = str(day_str).strip().lower()
        return day_mapping.get(clean_day, (0, 0))

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs data cleaning, formats columns, and engineers analytical features."""
        df_clean = df.copy()
        
        # Ensure column names are stripped of whitespace
        df_clean.columns = [col.strip() for col in df_clean.columns]
        
        # Clean counts and totals
        count_cols = ["CarCount", "BikeCount", "BusCount", "TruckCount"]
        for col in count_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0).astype(int)
            
        # Recompute total count to ensure mathematical alignment
        df_clean["Total"] = df_clean[count_cols].sum(axis=1)
        
        # Calculate engineered DensityScore
        df_clean["DensityScore"] = df_clean.apply(
            lambda row: self.calculate_density_score(
                row["CarCount"], row["BikeCount"], row["BusCount"], row["TruckCount"]
            ), axis=1
        )
        
        # Extract temporal features
        hours = []
        minutes = []
        peaks = []
        for time_val in df_clean["Time"]:
            h, m, p = self.parse_time_features(time_val)
            hours.append(h)
            minutes.append(m)
            peaks.append(p)
            
        df_clean["Hour"] = hours
        df_clean["Minute"] = minutes
        df_clean["IsPeakHour"] = peaks
        
        # Day of week features
        day_nums = []
        weekends = []
        for day in df_clean["Day of the week"]:
            num, wknd = self.encode_day_of_week(day)
            day_nums.append(num)
            weekends.append(wknd)
            
        df_clean["DayOfWeek"] = day_nums
        df_clean["IsWeekend"] = weekends
        
        # Clean target label situations
        df_clean["Traffic Situation"] = df_clean["Traffic Situation"].str.strip().str.lower()
        # Handle empty or invalid situations
        valid_situations = {"low", "normal", "high", "heavy"}
        df_clean["Traffic Situation"] = df_clean.apply(
            lambda r: r["Traffic Situation"] if r["Traffic Situation"] in valid_situations 
            else self.get_situation_from_score(r["DensityScore"]), axis=1
        )
        
        return df_clean

def run_data_pipeline():
    """Main pipeline execution for dataset ingestion and preprocessing."""
    logger.info("Initializing Data Ingestion & Preprocessing Pipeline...")
    
    # 1. Check for raw files and move/copy to datasets/raw
    raw_dir = Path(config["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    workspace_files = ["Traffic.csv", "TrafficTwoMonth.csv"]
    for file_name in workspace_files:
        src_path = Path(file_name)
        dest_path = raw_dir / file_name
        if src_path.exists() and not dest_path.exists():
            shutil.copy(src_path, dest_path)
            logger.info(f"Copied {file_name} to {dest_path}")
            
    # Check if we have files in raw_dir now
    raw_twomonth_path = raw_dir / "TrafficTwoMonth.csv"
    raw_month_path = raw_dir / "Traffic.csv"
    
    if not raw_twomonth_path.exists() or not raw_month_path.exists():
        logger.error(f"Missing raw CSV files in workspace. Run setup/verify datasets.")
        return False
        
    # 2. Process Datasets
    analyzer = TrafficDensityAnalyzer()
    
    logger.info("Loading and cleaning TrafficTwoMonth.csv (Training / Baseline dataset)...")
    df_twomonth = pd.read_csv(raw_twomonth_path)
    df_twomonth_clean = analyzer.clean_dataframe(df_twomonth)
    
    logger.info("Loading and cleaning Traffic.csv (Validation dataset)...")
    df_month = pd.read_csv(raw_month_path)
    df_month_clean = analyzer.clean_dataframe(df_month)
    
    # 3. Create Processed Folder and save cleaned datasets
    processed_dir = Path(config["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform train-test split on df_twomonth for machine learning training
    np.random.seed(config["ml_models"]["random_state"])
    shuffled_indices = np.random.permutation(len(df_twomonth_clean))
    test_size = int(len(df_twomonth_clean) * config["ml_models"]["test_size"])
    
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    
    df_train = df_twomonth_clean.iloc[train_indices]
    df_test = df_twomonth_clean.iloc[test_indices]
    
    # Save to processed CSVs
    df_train.to_csv(processed_dir / "train.csv", index=False)
    df_test.to_csv(processed_dir / "test.csv", index=False)
    df_month_clean.to_csv(processed_dir / "validation.csv", index=False)
    
    # Combine clean datasets for database ingestion
    combined_records = pd.concat([df_twomonth_clean, df_month_clean], ignore_index=True)
    # Drop duplicates by unique timestamp elements
    combined_records = combined_records.drop_duplicates(subset=["Time", "Date", "Day of the week"])
    
    logger.info(f"Processed datasets saved to {processed_dir}. Total training rows: {len(df_train)}, test rows: {len(df_test)}")
    
    # 4. Ingest into database
    db_manager = DatabaseManager()
    records_dict_list = combined_records.to_dict(orient="records")
    db_manager.insert_traffic_records_bulk(records_dict_list)
    logger.info("Data Engineering pipeline finished successfully.")
    return True

if __name__ == "__main__":
    run_data_pipeline()
