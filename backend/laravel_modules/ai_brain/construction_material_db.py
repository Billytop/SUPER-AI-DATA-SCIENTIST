
import random
from typing import Dict, Any

# SOVEREIGN CONSTRUCTION MATERIAL DATABASE v1.0
# Dynamic Cost Estimation for Building Materials.

class ConstructionMaterialDB:
    def __init__(self):
        self.materials = {
            "CEMENT_42.5N": {"base_price": 14000, "unit": "50kg Bag", "density": 1440}, # kg/m3
            "CONCRETE_C25": {"base_price": 280000, "unit": "m3", "density": 2400},
            "STEEL_Y12": {"base_price": 32000, "unit": "12m Bar", "density": 7850},
            "STEEL_Y16": {"base_price": 54000, "unit": "12m Bar", "density": 7850},
            "GLASS_TEMPERED": {"base_price": 85000, "unit": "m2", "density": 2500},
            "SAND_RIVER": {"base_price": 45000, "unit": "Ton", "density": 1600},
            "AGGREGATE": {"base_price": 55000, "unit": "Ton", "density": 1500},
            "PAINT_EMULSION": {"base_price": 65000, "unit": "20L Bucket", "density": 1200},
            "TILES_CERAMIC": {"base_price": 25000, "unit": "Box (1.44m2)", "density": 2000}
        }
        
        self.commodity_index = 1.0 # Market multiplier

    def update_market_prices(self, inflation_rate: float) -> None:
        """
        Simulates global market fluctuations.
        """
        self.commodity_index *= (1 + inflation_rate)
        
    def get_material_quote(self, material_key: str, quantity: float) -> Dict[str, Any]:
        """
        Calculates quote based on live market index.
        """
        mat = self.materials.get(material_key)
        if not mat:
            return {"error": "Material not found"}
            
        unit_price = mat['base_price'] * self.commodity_index
        total_cost = unit_price * quantity
        
        return {
            "material": material_key,
            "unit": mat['unit'],
            "qty": quantity,
            "unit_price_tzs": unit_price,
            "total_cost_tzs": total_cost,
            "market_index": self.commodity_index
        }

    def estimate_project_boq(self, sq_meters: float, floors: int) -> str:
        """
        Generates Bill of Quantities (BOQ) for a standard concrete structure.
        """
        # Heuristics:
        # Concrete: 0.4 m3 per m2 of floor area
        # Steel: 120 kg per m3 of concrete
        
        total_area = sq_meters * floors
        concrete_vol = total_area * 0.4
        steel_kg = concrete_vol * 120
        steel_bars_y16 = steel_kg / 18.9 # approx weight of Y16 bar
        
        concrete_cost = self.get_material_quote("CONCRETE_C25", concrete_vol)['total_cost_tzs']
        steel_cost = self.get_material_quote("STEEL_Y16", steel_bars_y16)['total_cost_tzs']
        
        total = concrete_cost + steel_cost
        
        return (
            f"### [PROJECT ESTIMATE: {floors} FLOORS]\n"
            f"- Total Floor Area: {total_area:,.0f} m2\n"
            f"- Concrete Required: {concrete_vol:,.0f} m3 ({concrete_cost/1e9:.2f}B TZS)\n"
            f"- Steel (Y16) Req: {steel_bars_y16:,.0f} bars ({steel_cost/1e9:.2f}B TZS)\n"
            f"- ESTIMATED SHELL COST: {total/1e9:.2f} BILLION TZS"
        )

MATERIAL_DB = ConstructionMaterialDB()
