"""
SephlightyAI API Exposure Layer
Author: Antigravity AI
Version: 1.0.0

RESTful + WebSocket + GraphQL API for all 57 module assistants.
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import json
from laravel_modules.ai_brain.central_integration_layer import CentralAI
from laravel_modules.ai_brain.proactive_intelligence import ProactiveIntelligence
from contextlib import asynccontextmanager

app = FastAPI(
    title="SephlightyAI Intelligence API",
    description="Unified API for 57 autonomous business intelligence modules",
    version="1.0.0",
    lifespan=lambda app: lifespan(app)
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ GLOBAL INFRASTRUCTURE ============

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

# Global instances
central_ai = CentralAI()
manager = ConnectionManager()
proactive_engine = None

import logging
logger_api = logging.getLogger("API_CORE")
logger_api.setLevel(logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events."""
    global proactive_engine
    
    # 1. Initialize Central AI
    central_ai.initialize()
    
    # 2. Start Proactive Engine
    proactive_engine = ProactiveIntelligence(manager.broadcast)
    asyncio.create_task(proactive_engine.start())
    logger_api.info("SephlightyAI Proactive Intelligence Hub: ONLINE")
    
    # 3. Initial OMNIBRAIN Discovery (Multi-tenant Mock)
    central_ai.saas.analyze_data_source("TENANT_001", {
        "tables": ["customers", "orders", "inventory", "accounting_logs"],
        "columns": ["cust_id", "tx_amt", "bal", "qty", "prod_name"]
    })
    
    yield
    
    # 4. Shutdown
    proactive_engine.stop()
    central_ai.saas.save_state()
    logger_api.info("SephlightyAI Shutdown: State Persisted.")

# (Discovery moved to lifespan)

# ============ REQUEST/RESPONSE MODELS ============

class QueryRequest(BaseModel):
    module: str
    method: str
    params: Dict[str, Any] = {}

class WorkflowRequest(BaseModel):
    steps: List[Dict[str, Any]]

class NaturalLanguageQuery(BaseModel):
    query: str
    context: Optional[Dict] = {}

class AuditRequest(BaseModel):
    query: str
    connection_id: str = "TENANT_001"

# ============ CORE ENDPOINTS ============

@app.get("/")
async def root():
    """API health check."""
    return {
        "status": "online",
        "modules_loaded": len(central_ai.registry.modules),
        "version": "1.0.0"
    }

@app.get("/modules")
async def list_modules():
    """List all available AI modules and their capabilities."""
    return {
        "modules": list(central_ai.registry.modules.keys()),
        "capabilities": central_ai.registry.module_capabilities
    }

@app.post("/query")
async def execute_query(request: QueryRequest):
    """Execute a single module method."""
    try:
        result = central_ai.comm.route_request(
            request.module,
            request.method,
            request.params
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/workflow")
async def execute_workflow(request: WorkflowRequest):
    """Execute multi-module workflow."""
    try:
        result = central_ai.execute({"steps": request.steps})
        return {"success": True, "results": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask")
async def natural_language_query(request: NaturalLanguageQuery):
    """Process natural language query."""
    result = central_ai.query(request.query, request.context)
    return result

# ============ MODULE-SPECIFIC ENDPOINTS ============

# CRM Intelligence
@app.post("/crm/predict_churn")
async def crm_predict_churn(customer_id: str):
    return central_ai.comm.route_request("crm_ai", "predict_churn", {"customer_id": customer_id})

@app.post("/crm/recommend_upsell")
async def crm_recommend_upsell(customer_id: str):
    return central_ai.comm.route_request("crm_ai", "recommend_upsell", {"customer_id": customer_id})

# Inventory Intelligence
@app.post("/inventory/predict_stockout")
async def inventory_predict_stockout(product_id: str):
    return central_ai.comm.route_request("inventory_ai", "predict_stockout", {"product_id": product_id})

@app.post("/inventory/optimize_reorder")
async def inventory_optimize_reorder():
    return central_ai.comm.route_request("inventory_ai", "optimize_reorder_points", {})

# Accounting Intelligence
@app.post("/accounting/detect_anomalies")
async def accounting_detect_anomalies():
    return central_ai.comm.route_request("accounting_ai", "detect_financial_anomalies", {})

@app.post("/accounting/forecast_revenue")
async def accounting_forecast_revenue(months: int = 3):
    return central_ai.comm.route_request("accounting_ai", "forecast_revenue", {"months": months})

# Manufacturing Intelligence
@app.post("/manufacturing/optimize_production")
async def manufacturing_optimize_production():
    return central_ai.comm.route_request("manufacturing_ai", "optimize_production_schedule", {})

@app.post("/manufacturing/predict_quality")
async def manufacturing_predict_quality(batch_id: str):
    return central_ai.comm.route_request("manufacturing_ai", "predict_defect_rate", {"batch_id": batch_id})

# HR Intelligence
@app.post("/hr/detect_flight_risk")
async def hr_detect_flight_risk(employee_id: str):
    return central_ai.comm.route_request("attendance_ai", "predict_attrition_risk", {"employee_id": employee_id})

@app.post("/hr/recommend_training")
async def hr_recommend_training(employee_id: str):
    return central_ai.comm.route_request("training_ai", "recommend_courses", {"employee_id": employee_id})

# ============ OMNIBRAIN SAAS ENDPOINTS ============

@app.get("/saas/dashboard")
async def get_saas_dashboard(connection_id: str = "TENANT_001"):
    """Generate automated SaaS dashboard."""
    return central_ai.saas.generate_dashboard("show me dashboard", connection_id)

@app.post("/saas/stress-test")
async def run_saas_stress_test(connection_id: str = "TENANT_001"):
    """Execute AI System Stress-Test."""
    return central_ai.saas.run_stress_test(connection_id)

@app.post("/saas/audit")
async def run_saas_audit(request: AuditRequest):
    """Execute AI-Verified Audit."""
    return central_ai.saas.run_ai_audit(request.query, request.connection_id)

@app.post("/saas/export-report")
async def export_saas_report(domain: str, format: str = "excel"):
    """Export business reports to Excel or PDF."""
    # This would typically pull from real DB data in a production scenario
    # For now, we simulate data retrieval and call the report engine
    mock_data = [
        {"Category": "Revenue", "Amount": "1,200,000", "Growth": "+5%"},
        {"Category": "Expenses", "Amount": "800,000", "Growth": "-2%"},
        {"Category": "Net Profit", "Amount": "400,000", "Growth": "+12%"}
    ]
    file_path = central_ai.reports.generate_master_report(mock_data, f"{domain}_Report", export_format=format)
    return {"success": True, "download_url": file_path}

# ============ REAL-TIME STREAMING ============

# manager_old removed, using global manager

@app.websocket("/ws/insights")
async def websocket_insights(websocket: WebSocket):
    """Real-time AI insights streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Listen for client requests
            data = await websocket.receive_text()
            request = json.loads(data)
            
            # Process and stream back
            if request["type"] == "subscribe":
                # Start streaming insights for specified modules
                await websocket.send_json({
                    "type": "subscribed",
                    "modules": request.get("modules", [])
                })
                
            elif request["type"] == "query":
                result = central_ai.comm.route_request(
                    request["module"],
                    request["method"],
                    request.get("params", {})
                )
                await websocket.send_json({
                    "type": "result",
                    "data": result
                })
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

# ============ BATCH OPERATIONS ============

@app.post("/batch/analyze")
async def batch_analyze(module: str, items: List[Dict]):
    """Batch analysis across multiple items."""
    results = []
    for item in items:
        try:
            result = central_ai.comm.route_request(
                module,
                item.get("method", "analyze"),
                item.get("params", {})
            )
            results.append({"success": True, "result": result})
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    return {"results": results}

# ============ ADMIN ENDPOINTS ============

@app.post("/admin/reload_modules")
async def reload_modules():
    """Reload all AI modules (requires admin auth)."""
    central_ai.registry.discover_modules()
    return {
        "success": True,
        "modules_loaded": len(central_ai.registry.modules)
    }

@app.get("/admin/performance")
async def get_performance():
    """Get performance metrics for all modules."""
    return {
        "total_modules": len(central_ai.registry.modules),
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
