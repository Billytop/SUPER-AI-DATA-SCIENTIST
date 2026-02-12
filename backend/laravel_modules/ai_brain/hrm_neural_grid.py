
import random
from typing import Dict, List, Any

# SOVEREIGN HRM NEURAL GRID v2.5
# Employee Analytics: Burnout Prediction, Performance Scoring, and Payroll Optimization.

class HRMNeuralEngine:
    def __init__(self):
        self.burnout_threshold = 0.8
        
    def analyze_employee_burnout(self, shift_hours: List[float], overtime_hours: List[float]) -> str:
        """
        Uses neural heuristics to predict if an employee is about to quit or underperform due to burnout.
        """
        total_hours = sum(shift_hours) + sum(overtime_hours)
        avg_daily = total_hours / len(shift_hours) if shift_hours else 0
        
        # Burnout Score Calculation
        score = 0.0
        if avg_daily > 10: score += 0.4
        if avg_daily > 12: score += 0.3
        if sum(overtime_hours) > 20: score += 0.2
        
        if score > self.burnout_threshold:
            return "CRITICAL BURNOUT RISK: Rest Day Mandatory"
        elif score > 0.5:
            return "HIGH STRESS: Monitor Workload"
        else:
            return "OPTIMAL PERFORMANCE MODE"

    def calculate_performance_score(self, sales: float, errors: int, attendance: float) -> float:
        """
        360-Degree Performance Review Algorithm.
        """
        # Weighted factors
        w_sales = 0.5
        w_errors = -0.3
        w_attendance = 0.2
        
        # Normalize (assuming target sales 1M)
        norm_sales = min(sales / 1000000.0, 1.5)
        norm_errors = min(errors / 10.0, 1.0)
        norm_attendance = min(attendance / 100.0, 1.0)
        
        final_score = (norm_sales * w_sales) + (norm_errors * w_errors) + (norm_attendance * w_attendance)
        
        # Scale to 0-100
        return max(min(final_score * 100, 100.0), 0.0)

    def optimize_shift_schedule(self, employees: List[str], expected_demand: str) -> List[str]:
        """
        AI-driven shift allocation based on predicted foot traffic.
        """
        if expected_demand == "HIGH":
            return employees # All hands on deck
        elif expected_demand == "MEDIUM":
            return employees[:int(len(employees)*0.7)]
        else:
            return employees[:int(len(employees)*0.4)]

HRM_NEURAL = HRMNeuralEngine()
