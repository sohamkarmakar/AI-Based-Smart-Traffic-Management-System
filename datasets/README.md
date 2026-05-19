# Traffic Prediction Dataset

This directory contains the raw and processed datasets used to train and evaluate models for congestion prediction and density analysis.

## Dataset Structure

```
datasets/
│
├── raw/                 # Unmodified, original CSV files (Traffic.csv, TrafficTwoMonth.csv)
├── processed/           # Processed datasets ready for model ingestion
│   ├── train.csv        # 80% train split (from TrafficTwoMonth.csv)
│   ├── test.csv         # 20% test split (from TrafficTwoMonth.csv)
│   └── validation.csv   # Validation set (from Traffic.csv)
│
└── README.md            # This documentation file
```

## Setup Instructions

1. Put the original Kaggle CSV files (`Traffic.csv` and `TrafficTwoMonth.csv`) into the `datasets/raw/` folder, or let the data ingestion pipeline do it automatically on startup.
2. Run the ingestion and preprocessing script:
   ```bash
   python -m src.analytics.density_analyzer
   ```
   This will clean the data, generate engineered time features, compute weighted density scores, split the data, and output the processed datasets under `datasets/processed/`.

## Data Dictionary

| Column Name | Data Type | Description | Values / Range |
|---|---|---|---|
| `Time` | `string` | 15-minute interval start timestamp | e.g. `12:15:00 AM`, `04:30:00 PM` |
| `Date` | `integer` | Day of the month | `1` to `31` |
| `Day of the week` | `string` | Day of week | `Monday` - `Sunday` |
| `CarCount` | `integer` | Count of cars observed | `0` to `200+` |
| `BikeCount` | `integer` | Count of motorbikes/bicycles | `0` to `100+` |
| `BusCount` | `integer` | Count of passenger buses | `0` to `50+` |
| `TruckCount` | `integer` | Count of logistics/commercial trucks | `0` to `50+` |
| `Total` | `integer` | Simple sum of all vehicle counts | `0` to `300+` |
| `Traffic Situation` | `string` | Human-labelled congestion class | `low`, `normal`, `high`, `heavy` |
| `DensityScore` | `float` (engineered) | Weighted road occupancy score (PCU-equivalent) | `0.0` to `150.0+` |
| `Hour` | `integer` (engineered) | Hour of the day | `0` to `23` |
| `DayOfWeek` | `integer` (engineered) | Encoded day of week | `0` (Monday) to `6` (Sunday) |
| `IsWeekend` | `integer` (engineered) | Binary indicator for Saturday/Sunday | `0` or `1` |
| `IsPeakHour` | `integer` (engineered) | Binary indicator for rush-hour peak times | `0` or `1` |

## Data Preprocessing Notes
- **Outliers**: Outliers are not deleted since heavy traffic spikes contain crucial signals for predicting real-world gridlocks.
- **Weighted Scores**: We calculate the `DensityScore` using standard road passenger car unit (PCU) values: Cars = 1.0, Bikes = 0.2, Buses = 2.5, Trucks = 3.0.
