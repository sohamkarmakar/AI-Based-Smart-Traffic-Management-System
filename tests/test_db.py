import unittest
import os
from pathlib import Path
from src.database.db_manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    """Unit tests for the SQLite Database Manager class."""

    def setUp(self):
        # Use a temporary test database file path
        self.test_db_path = "outputs/test_traffic_management.db"
        self.db = DatabaseManager(db_path=self.test_db_path)
        self.db.clear_all_records()

    def tearDown(self):
        # Remove test database file
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_database_initialization(self):
        """Verifies that database manager sets up the three tables correctly."""
        self.assertTrue(os.path.exists(self.test_db_path))

    def test_insert_and_query_traffic_records(self):
        """Verifies insertion and retrieval of telemetry records."""
        sample_records = [
            {
                "Time": "08:30:00 AM",
                "Date": 10,
                "Day of the week": "Tuesday",
                "CarCount": 50,
                "BikeCount": 10,
                "BusCount": 4,
                "TruckCount": 2,
                "Total": 66,
                "DensityScore": 68.0,
                "Traffic Situation": "normal"
            }
        ]
        self.db.insert_traffic_records_bulk(sample_records)
        history = self.db.get_historical_records(limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["day_of_week"], "Tuesday")
        self.assertEqual(history[0]["total_volume"], 66)

    def test_insert_prediction(self):
        """Verifies insertion of prediction audit logs."""
        self.db.insert_prediction(
            prediction_time="2026-05-19 20:00:00",
            input_features={"CarCount": 10},
            predicted_situation="low",
            predicted_volume=12.0,
            model_type="xgboost"
        )
        preds = self.db.get_recent_predictions(limit=5)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["predicted_situation"], "low")
        self.assertEqual(preds[0]["input_features"]["CarCount"], 10)

if __name__ == "__main__":
    unittest.main()
