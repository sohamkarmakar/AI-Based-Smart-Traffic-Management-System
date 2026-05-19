import unittest
from src.analytics.density_analyzer import TrafficDensityAnalyzer

class TestTrafficDensityAnalyzer(unittest.TestCase):
    """Unit tests for the Traffic Density Analyzer feature extraction."""

    def setUp(self):
        self.analyzer = TrafficDensityAnalyzer()

    def test_pcu_calculation(self):
        """Verifies Passenger Car Unit score weighting logic."""
        # weights: car=1.0, bike=0.2, bus=2.5, truck=3.0
        # 10 cars, 5 bikes, 2 buses, 1 truck
        # Score = 10*1.0 + 5*0.2 + 2*2.5 + 1*3.0 = 10 + 1 + 5 + 3 = 19.0
        score = self.analyzer.calculate_density_score(10, 5, 2, 1)
        self.assertEqual(score, 19.0)

    def test_situation_from_score(self):
        """Ensures density score ranges correctly map to situation classes."""
        self.assertEqual(self.analyzer.get_situation_from_score(15.0), "low")
        self.assertEqual(self.analyzer.get_situation_from_score(50.0), "normal")
        self.assertEqual(self.analyzer.get_situation_from_score(90.0), "high")
        self.assertEqual(self.analyzer.get_situation_from_score(140.0), "heavy")

    def test_time_parsing(self):
        """Verifies AM/PM strings are parsed correctly into hour/minute and peak hour status."""
        # 08:30 AM is peak hour (7:00 - 9:30 AM)
        h, m, p = self.analyzer.parse_time_features("08:30:00 AM")
        self.assertEqual(h, 8)
        self.assertEqual(m, 30)
        self.assertEqual(p, 1)

        # 11:00 AM is off-peak
        h, m, p = self.analyzer.parse_time_features("11:00:00 AM")
        self.assertEqual(h, 11)
        self.assertEqual(m, 0)
        self.assertEqual(p, 0)

        # 05:15 PM is peak hour (4:30 - 7:30 PM)
        h, m, p = self.analyzer.parse_time_features("05:15:00 PM")
        self.assertEqual(h, 17)
        self.assertEqual(m, 15)
        self.assertEqual(p, 1)

if __name__ == "__main__":
    unittest.main()
