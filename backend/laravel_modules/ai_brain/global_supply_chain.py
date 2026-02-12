import logging
import math
import random
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class SovereignSupplyChain:
    """
    GLOBAL SUPPLY CHAIN & ARBITRAGE ENGINE v4.0
    Autonomous logistics optimization, detailed route planning, and 
    cross-border price arbitrage detection.
    """

    def __init__(self):
        self.routes = {
            "mombasa_nairobi": {"distance_km": 480, "avg_speed_kmh": 60, "road_tax": 5000},
            "dar_mwanza": {"distance_km": 1150, "avg_speed_kmh": 55, "road_tax": 12000},
            "dar_arusha": {"distance_km": 650, "avg_speed_kmh": 70, "road_tax": 7500},
            "kampala_nairobi": {"distance_km": 660, "avg_speed_kmh": 50, "road_tax": 15000}
        }
        
        self.fuel_price_per_liter = 3200 # TZS
        self.truck_efficiency_kml = 3.5 # Km per Liter

    def calculate_freight_cost(self, route_key: str, cargo_weight_tons: float) -> Dict[str, float]:
        """Calculates detailed logistics cost including fuel, tax, and driver allowance."""
        route = self.routes.get(route_key)
        if not route:
            return {"error": "Route not mapped in Sovereign GPS."}
            
        distance = route['distance_km']
        fuel_needed = distance / self.truck_efficiency_kml
        fuel_cost = fuel_needed * self.fuel_price_per_liter
        
        # Heavy load penalty
        if cargo_weight_tons > 20:
            fuel_cost *= 1.2 # 20% more fuel for heavy loads
            
        driver_allowance = (distance / route['avg_speed_kmh']) * 5000 # 5000 TZS per hour
        total_cost = fuel_cost + route['road_tax'] + driver_allowance
        
        return {
            "fuel_cost": fuel_cost,
            "road_tax": route['road_tax'],
            "driver_allowance": driver_allowance,
            "total_trip_cost": total_cost,
            "cost_per_km": total_cost / distance,
            "cost_per_ton": total_cost / cargo_weight_tons
        }

    def detect_arbitrage(self, product: str, markets: List[Dict[str, float]]) -> List[Dict]:
        """
        Identifies profitable trade routes between markest based on price differences
        minus transport costs.
        markets format: [{'name': 'Dar', 'price': 5000}, {'name': 'Arusha', 'price': 6500}]
        """
        opportunities = []
        
        for source in markets:
            for dest in markets:
                if source['name'] == dest['name']: continue
                
                # Check if route exists
                route_key = f"{source['name'].lower()}_{dest['name'].lower()}"
                # Try simple heuristic for reverse route or direct map
                if route_key not in self.routes:
                    # Mock distance for unmapped
                    transport_cost = 200 * 500 # Assume 500km avg * 200 TZS/unit transport
                else:
                    # Precise calc
                    cost_data = self.calculate_freight_cost(route_key, 10) # Assume 10 ton batch
                    transport_cost = cost_data['cost_per_ton'] / 1000 # Cost per kg approximation
                    
                gross_margin = dest['price'] - source['price']
                net_margin = gross_margin - transport_cost
                
                if net_margin > 0:
                    opportunities.append({
                        "route": f"{source['name']} -> {dest['name']}",
                        "buy_at": source['price'],
                        "sell_at": dest['price'],
                        "transport_cost_unit": transport_cost,
                        "net_profit_unit": net_margin,
                        "roi": (net_margin / source['price']) * 100
                    })
                    
        return sorted(opportunities, key=lambda x: x['net_profit_unit'], reverse=True)

    def optimize_warehouse_location(self, demand_centers: List[Dict[str, float]]) -> str:
        """
        Uses Center of Gravity method to suggest optimal warehouse location.
        demand_centers: [{'x': 10, 'y': 20, 'volume': 500}]
        """
        total_vol = sum(d['volume'] for d in demand_centers)
        weighted_x = sum(d['x'] * d['volume'] for d in demand_centers) / total_vol
        weighted_y = sum(d['y'] * d['volume'] for d in demand_centers) / total_vol
        
        return f"Optimal Hub Coordinates: ({weighted_x:.2f}, {weighted_y:.2f}). Suggest proximity to nearest highway interchange."

GLOBAL_SUPPLY = SovereignSupplyChain()
