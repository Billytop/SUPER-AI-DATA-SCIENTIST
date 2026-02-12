
import random
from typing import Dict, Any

# SOVEREIGN SMART CITY GRID SIMULATOR v2.0
# Power & Water Consumption Logic for Urban Planning.

class SmartCityGrid:
    def __init__(self):
        self.grid_capacity_mw = 500 # Megawatts
        self.water_reserves_m3 = 10000000 # Cubic Meters
        
    def calculate_building_load(self, building_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Estimates daily utility consumption for a structure.
        """
        if building_data['type'] == "SKYSCRAPER":
            # 100 kWh per floor per day (Base) + HVAC load
            base_power = building_data['total_floors'] * 150 
            hvac_load = building_data['footprint'] * 0.05 # kWh
            total_kwh = base_power + hvac_load
            
            # Water: 200 Liters per 10 sqm (Occupancy)
            water_liters = building_data['footprint'] * building_data['total_floors'] * 20
            
            return {
                "power_daily_kwh": total_kwh,
                "water_daily_m3": water_liters / 1000,
                "peak_load_kw": total_kwh / 12 # Assume 12h active window
            }
            
        elif building_data['type'] == "MEGA_MALL":
            # Malls consume huge power for lighting/AC
            total_shops = building_data['total_shops']
            power = total_shops * 50 # kWh per shop
            water = total_shops * 500 # Liters per shop (Food courts etc)
            
            return {
                "power_daily_kwh": power,
                "water_daily_m3": water / 1000,
                "peak_load_kw": power / 10
            }
            
        return {"power_daily_kwh": 0, "water_daily_m3": 0}

    def grid_stability_check(self, current_load_mw: float) -> str:
        """
        Checks if the city grid can handle the load.
        """
        usage_percent = (current_load_mw / self.grid_capacity_mw) * 100
        
        if usage_percent > 90:
            return "CRITICAL ALERT: Grid Overload Imminent (Load Shedding Required)"
        elif usage_percent > 75:
            return "WARNING: High Demand on Grid"
        else:
            return "GRID STABLE: Capacity Available"

CITY_GRID = SmartCityGrid()
