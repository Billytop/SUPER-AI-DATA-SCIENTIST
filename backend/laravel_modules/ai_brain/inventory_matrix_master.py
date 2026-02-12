
import datetime
import random
from typing import Dict, List, Any

# SOVEREIGN INVENTORY MATRIX MASTER v3.0
# Advanced Stock Management: Dead Stock, Shrinkage, Expiry.

class InventoryMatrixEngine:
    def __init__(self):
        self.disposal_threshold_days = 180 # 6 months
        self.shrinkage_alert_limit = 0.05 # 5% discrepancy
        
    def analyze_stock_velocity(self, stock_in_date: datetime.date, sales_velocity: float) -> str:
        """
        Determines if an item is fast-moving, slow-moving, or dead stock.
        """
        days_on_shelf = (datetime.date.today() - stock_in_date).days
        
        if sales_velocity > 10.0:
            return "FAST MOVER: Reorder Immediately"
        elif sales_velocity > 2.0:
            return "NORMAL MOVER: Monitor Levels"
        elif days_on_shelf > self.disposal_threshold_days:
            return "DEAD STOCK: Liquidation Recommended (Discount 50%)"
        else:
            return "SLOW MOVER: Consider Promotion"

    def detect_shrinkage(self, system_count: int, physical_count: int) -> Dict[str, Any]:
        """
        Compares system vs physical stock to detect theft or loss.
        """
        discrepancy = system_count - physical_count
        shrinkage_rate = discrepancy / system_count if system_count > 0 else 0
        
        status = "NORMAL"
        action = "None"
        
        if shrinkage_rate > self.shrinkage_alert_limit:
            status = "CRITICAL SHRINKAGE"
            action = "Audit Required & Check CCTV"
        elif shrinkage_rate > 0:
            status = "MINOR LOSS"
            action = "Monitor Closely"
        elif shrinkage_rate < 0:
            status = "SURPLUS (Error)"
            action = "Retrain Staff on Entry"
            
        return {
            "system": system_count,
            "physical": physical_count,
            "missing": discrepancy,
            "rate": f"{shrinkage_rate*100:.2f}%",
            "status": status,
            "action": action
        }

    def optimize_expiry_dispatch(self, batch_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Implements FEFO (First Expiry First Out) logic.
        Sorts batches by expiry date.
        """
        sorted_batches = sorted(batch_list, key=lambda x: x['expiry_date'])
        
        for batch in sorted_batches:
            days_left = (batch['expiry_date'] - datetime.date.today()).days
            if days_left < 30:
                batch['priority'] = "URGENT DISPATCH"
                batch['discount'] = 0.25
            elif days_left < 90:
                batch['priority'] = "HIGH PRIORITY"
                batch['discount'] = 0.10
            else:
                batch['priority'] = "NORMAL"
                batch['discount'] = 0.0
                
        return sorted_batches

    def generate_receipt_fraud_score(self, receipt_data: Dict[str, Any]) -> float:
        """
        Calculates likelihood of receipt manipulation.
        """
        score = 0.0
        # Pseudo-heuristic checks
        if "reprint_count" in receipt_data and receipt_data["reprint_count"] > 2:
            score += 0.4
        if "void_items" in receipt_data and len(receipt_data["void_items"]) > 3:
            score += 0.3
        if "timestamp" in receipt_data and receipt_data["timestamp"].hour < 6: # Middle of night
            score += 0.2
            
        return min(score, 1.0)

INVENTORY_MATRIX = InventoryMatrixEngine()
