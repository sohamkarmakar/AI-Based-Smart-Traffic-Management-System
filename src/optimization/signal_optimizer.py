import os
from datetime import datetime
from src.utils.logger import logger
from src.utils.config_loader import config
from src.database.db_manager import DatabaseManager

class TrafficSignalOptimizer:
    """Intelligent recommendation engine for dynamic traffic signal phase durations."""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager if db_manager is not None else DatabaseManager()
        self.min_green = config["signal_optimization"]["min_green_time"]
        self.max_green = config["signal_optimization"]["max_green_time"]
        self.yellow = config["signal_optimization"]["base_yellow_time"]
        self.red = config["signal_optimization"]["base_red_time"]
        self.pcu_flow_rate = config["signal_optimization"]["pcu_per_second"]

    def optimize_signal_timing(self, lanes_data: dict[str, dict]) -> dict:
        """Calculates optimal green phase durations for a 4-way intersection.
        
        Args:
            lanes_data: Dict of lane measurements:
                        {
                           'north': {'car': 40, 'bike': 10, 'bus': 5, 'truck': 2},
                           'south': {'car': 15, 'bike': 2, 'bus': 1, 'truck': 0},
                           ...
                        }
        Returns:
            optimization_results: dict containing timings, sequence, priority and savings.
        """
        results = {}
        lane_densities = {}
        total_pcu = 0.0
        
        # Calculate densities (PCU equivalents) for all approaches
        weights = config["density_analysis"]["vehicle_weights"]
        for lane, counts in lanes_data.items():
            cars = counts.get("car", 0)
            bikes = counts.get("bike", 0)
            buses = counts.get("bus", 0)
            trucks = counts.get("truck", 0)
            
            pcu = (
                cars * weights["car"] +
                bikes * weights["bike"] +
                buses * weights["bus"] +
                trucks * weights["truck"]
            )
            lane_densities[lane] = round(pcu, 2)
            total_pcu += pcu

        # Determine green time per lane
        # Heuristic: base green + proportional scale up to max_green for highest density (reference 100 PCU gridlock)
        REFERENCE_MAX_PCU = 80.0
        recommended_green = {}
        estimated_wait_dynamic = 0.0
        estimated_wait_fixed = 0.0
        
        for lane, pcu in lane_densities.items():
            if pcu == 0:
                green_time = self.min_green
            else:
                proportion = min(1.0, pcu / REFERENCE_MAX_PCU)
                green_time = int(self.min_green + proportion * (self.max_green - self.min_green))
            recommended_green[lane] = green_time
            
            # Simple queuing heuristic:
            # Fixed cycle: 30s green. If PCU volume exceeds clearing capacity (pcu_flow_rate * green), queue builds up.
            cleared_fixed = 30 * self.pcu_flow_rate
            remaining_fixed = max(0.0, pcu - cleared_fixed)
            wait_fixed = (pcu * 15) + (remaining_fixed * 45) # delay penalization
            estimated_wait_fixed += wait_fixed
            
            cleared_dyn = green_time * self.pcu_flow_rate
            remaining_dyn = max(0.0, pcu - cleared_dyn)
            wait_dyn = (pcu * (green_time / 2.0)) + (remaining_dyn * 45)
            estimated_wait_dynamic += wait_dyn

        # Sort approaches by density to define priority scheduling
        sorted_lanes = sorted(lane_densities.items(), key=lambda x: x[1], reverse=True)
        priority_lane = sorted_lanes[0][0]
        priority_pcu = sorted_lanes[0][1]
        
        # Priority order
        priority_sequence = [lane for lane, _ in sorted_lanes]
        
        # Recommendations
        congestion_reduction_est = 0.0
        if estimated_wait_fixed > 0:
            congestion_reduction_est = round(((estimated_wait_fixed - estimated_wait_dynamic) / estimated_wait_fixed) * 100.0, 1)
            congestion_reduction_est = max(0.0, congestion_reduction_est)

        action_recommendations = []
        if priority_pcu > 60.0:
            action_recommendations.append(f"Deploy Peak Extension on {priority_lane.capitalize()} lane (+{recommended_green[priority_lane] - 30}s green time).")
            action_recommendations.append("Restrict heavy freight trucks from scheduling transit in peak phases.")
        elif priority_pcu > 30.0:
            action_recommendations.append(f"Increase green duration slightly on {priority_lane.capitalize()} lane.")
        else:
            action_recommendations.append("Standard signal cycles are sufficient. Recommend energy-saving mode (shorter green intervals).")

        results = {
            "lane_densities": lane_densities,
            "recommended_green_times": recommended_green,
            "priority_lane": priority_lane,
            "priority_sequence": priority_sequence,
            "estimated_delay_reduction_pct": congestion_reduction_est,
            "actions": action_recommendations,
            "cycle_metrics": {
                "dynamic_cycle_time": sum(recommended_green.values()) + len(recommended_green) * (self.yellow + self.red),
                "fixed_cycle_time": 4 * (30 + self.yellow + self.red)
            }
        }
        
        # Log to Database
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db_manager.insert_signal_recommendation(
            timestamp=now_str,
            lane_data=lanes_data,
            recommended_green_times=recommended_green,
            estimated_delay_reduction=congestion_reduction_est
        )
        
        return results
