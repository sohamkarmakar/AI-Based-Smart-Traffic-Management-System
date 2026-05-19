import json
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from src.utils.logger import logger
from src.utils.config_loader import config

class DatabaseManager:
    """Handles SQLite database storage for telemetry logs, predictions, and signal recommendations."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = config["database"]["db_path"]
        
        self.db_path = Path(db_path)
        # Ensure outputs folder exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _connection_context(self):
        """Context manager that opens and guarantees closing of SQLite connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        """Initializes database schema if tables do not exist."""
        schema_queries = [
            """
            CREATE TABLE IF NOT EXISTS traffic_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date_day INTEGER NOT NULL,
                day_of_week TEXT NOT NULL,
                car_count INTEGER NOT NULL,
                bike_count INTEGER NOT NULL,
                bus_count INTEGER NOT NULL,
                truck_count INTEGER NOT NULL,
                total_volume INTEGER NOT NULL,
                density_score REAL NOT NULL,
                traffic_situation TEXT NOT NULL,
                UNIQUE(timestamp, date_day, day_of_week) ON CONFLICT REPLACE
            );
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_records_timestamp ON traffic_records(timestamp);
            """,
            """
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time TEXT NOT NULL,
                input_features TEXT NOT NULL,  -- JSON string of inputs
                predicted_situation TEXT,
                predicted_volume REAL,
                model_type TEXT NOT NULL,
                confidence_score REAL
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS signal_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                lane_data TEXT NOT NULL,       -- JSON string of lane counts/densities
                recommended_green_times TEXT NOT NULL, -- JSON string of light phases
                estimated_delay_reduction REAL
            );
            """
        ]
        
        try:
            with self._connection_context() as conn:
                cursor = conn.cursor()
                for query in schema_queries:
                    cursor.execute(query)
                conn.commit()
            logger.info(f"Database initialized successfully at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise e

    def insert_traffic_records_bulk(self, records: list[dict]):
        """Inserts a batch of traffic records into the database."""
        query = """
            INSERT OR REPLACE INTO traffic_records (
                timestamp, date_day, day_of_week, car_count, bike_count, 
                bus_count, truck_count, total_volume, density_score, traffic_situation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        data_tuples = [
            (
                r.get("Time"),
                int(r.get("Date")),
                r.get("Day of the week"),
                int(r.get("CarCount", 0)),
                int(r.get("BikeCount", 0)),
                int(r.get("BusCount", 0)),
                int(r.get("TruckCount", 0)),
                int(r.get("Total", 0)),
                float(r.get("DensityScore", 0.0)),
                r.get("Traffic Situation", "normal")
            )
            for r in records
        ]
        
        try:
            with self._connection_context() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, data_tuples)
                conn.commit()
            logger.info(f"Bulk inserted {len(records)} records into traffic_records.")
        except sqlite3.Error as e:
            logger.error(f"Failed bulk insertion of traffic records: {e}")
            raise e

    def insert_prediction(self, prediction_time: str, input_features: dict, 
                          predicted_situation: str, predicted_volume: float, 
                          model_type: str, confidence_score: float = None):
        """Logs a single model prediction run to database for audit history."""
        query = """
            INSERT INTO model_predictions (
                prediction_time, input_features, predicted_situation, 
                predicted_volume, model_type, confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            with self._connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (
                    prediction_time,
                    json.dumps(input_features),
                    predicted_situation,
                    predicted_volume,
                    model_type,
                    confidence_score
                ))
                conn.commit()
            logger.info("Successfully logged traffic prediction to SQLite.")
        except sqlite3.Error as e:
            logger.error(f"Failed to log prediction: {e}")

    def insert_signal_recommendation(self, timestamp: str, lane_data: dict, 
                                     recommended_green_times: dict, estimated_delay_reduction: float):
        """Logs a single signal optimizer recommendation run to database."""
        query = """
            INSERT INTO signal_recommendations (
                timestamp, lane_data, recommended_green_times, estimated_delay_reduction
            ) VALUES (?, ?, ?, ?)
        """
        try:
            with self._connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (
                    timestamp,
                    json.dumps(lane_data),
                    json.dumps(recommended_green_times),
                    estimated_delay_reduction
                ))
                conn.commit()
            logger.info("Successfully logged traffic signal recommendation to SQLite.")
        except sqlite3.Error as e:
            logger.error(f"Failed to log signal recommendation: {e}")

    def get_historical_records(self, limit: int = 10000, day_of_week: str = None, 
                               min_density: float = None) -> list[dict]:
        """Fetches historical records based on query filters."""
        query = "SELECT * FROM traffic_records WHERE 1=1"
        params = []
        
        if day_of_week:
            query += " AND day_of_week = ?"
            params.append(day_of_week)
        
        if min_density is not None:
            query += " AND density_score >= ?"
            params.append(min_density)
            
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        try:
            with self._connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch historical records: {e}")
            return []

    def get_recent_predictions(self, limit: int = 10) -> list[dict]:
        """Fetches recent model predictions."""
        query = "SELECT * FROM model_predictions ORDER BY id DESC LIMIT ?"
        try:
            with self._connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    item = dict(row)
                    item["input_features"] = json.loads(item["input_features"])
                    results.append(item)
                return results
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch predictions: {e}")
            return []

    def get_recent_signal_recommendations(self, limit: int = 10) -> list[dict]:
        """Fetches recent signal timing optimization recommendations."""
        query = "SELECT * FROM signal_recommendations ORDER BY id DESC LIMIT ?"
        try:
            with self._connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()
                results = []
                for row in rows:
                    item = dict(row)
                    item["lane_data"] = json.loads(item["lane_data"])
                    item["recommended_green_times"] = json.loads(item["recommended_green_times"])
                    results.append(item)
                return results
        except sqlite3.Error as e:
            logger.error(f"Failed to fetch signal recommendations: {e}")
            return []

    def clear_all_records(self):
        """Clears all records from tables. Useful for unit testing."""
        try:
            with self._connection_context() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM traffic_records")
                cursor.execute("DELETE FROM model_predictions")
                cursor.execute("DELETE FROM signal_recommendations")
                conn.commit()
            logger.info("Cleared all records in SQLite database.")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear database records: {e}")
            raise e
