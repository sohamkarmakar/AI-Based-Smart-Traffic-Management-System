import unittest
import os
from src.database.db_manager import DatabaseManager
from src.optimization.signal_optimizer import TrafficSignalOptimizer

class TestTrafficSignalOptimizer(unittest.TestCase):
    """Unit tests for the Intelligent Traffic Signal Optimizer class."""

    def setUp(self):
        self.test_db_path = "outputs/test_signal_opt.db"
        self.db = DatabaseManager(db_path=self.test_db_path)
        self.optimizer = TrafficSignalOptimizer(db_manager=self.db)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_optimize_signal_timings_bounds(self):
        """Ensures optimized green times stay strictly within configured min/max limits."""
        lanes_data = {
            "north": {"car": 10, "bike": 2, "bus": 0, "truck": 0},
            "south": {"car": 200, "bike": 40, "bus": 25, "truck": 15}, # Heavy volume
            "east": {"car": 0, "bike": 0, "bus": 0, "truck": 0},
            "west": {"car": 45, "bike": 5, "bus": 2, "truck": 1}
        }
        
        results = self.optimizer.optimize_signal_timing(lanes_data)
        green_times = results["recommended_green_times"]
        
        # Test boundaries
        for lane, seconds in green_times.items():
            self.assertTrue(15 <= seconds <= 90, f"Green duration {seconds}s for {lane} out of bounds.")

    def test_priority_lane_selection(self):
        """Verifies that the lane with the highest density is selected as priority."""
        lanes_data = {
            "north": {"car": 10, "bike": 0, "bus": 0, "truck": 0},
            "south": {"car": 12, "bike": 0, "bus": 0, "truck": 0},
            "east": {"car": 150, "bike": 0, "bus": 0, "truck": 0}, # Highest density
            "west": {"car": 5, "bike": 0, "bus": 0, "truck": 0}
        }
        results = self.optimizer.optimize_signal_timing(lanes_data)
        self.assertEqual(results["priority_lane"], "east")

if __name__ == "__main__":
    unittest.main()
