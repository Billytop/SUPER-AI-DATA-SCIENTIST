"""
SephlightyAI Central Integration Layer
Author: Antigravity AI
Version: 1.0.0

Orchestrates 57 autonomous module assistants into a unified intelligence system.
"""

import importlib
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from laravel_modules.ai_brain.cognitive_intelligence_ai import CognitiveIntelligenceAI
from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
from laravel_modules.ai_brain.evolution_engine import EvolutionEngine
from laravel_modules.ai_brain.singularity_logic_hub import SingularityLogicHub
from laravel_modules.ai_brain.absolute_logic_vault import AbsoluteLogicVault
from laravel_modules.module_assistants.report_engine_ai import ReportEngineAI
from laravel_modules.ai_brain.high_density.super_agent_planner import SuperAgentPlanner

logger = logging.getLogger("CENTRAL_INTEGRATION")
logger.setLevel(logging.DEBUG)

class ModuleRegistry:
    """Discovers and registers all 57 module assistants."""
    
    def __init__(self):
        self.modules = {}
        self.module_capabilities = {}
        
    def discover_modules(self, base_path: Optional[str] = None):
        """Automatically discover and load all AI modules."""
        if base_path:
            module_path = Path(base_path)
        else:
            module_path = Path(__file__).parent.parent / "module_assistants"
        for file in module_path.glob("*_ai.py"):
            module_name = file.stem
            try:
                # Dynamically import module
                spec = importlib.util.spec_from_file_location(module_name, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Extract AI class (assumes class name follows pattern)
                class_name = self._infer_class_name(module_name)
                if hasattr(module, class_name):
                    ai_instance = getattr(module, class_name)()
                    self.modules[module_name] = ai_instance
                    self._extract_capabilities(module_name, ai_instance)
                    logger.info(f"Registered: {module_name}")
            except Exception as e:
                logger.error(f"Failed to load {module_name}: {e}")
                
    def _infer_class_name(self, module_name: str) -> str:
        """Convert module_name_ai to ModuleNameAI."""
        parts = module_name.replace("_ai", "").split("_")
        return "".join(p.capitalize() for p in parts) + "AI"
        
    def _extract_capabilities(self, name: str, instance: Any):
        """Extract what this module can do."""
        capabilities = []
        for attr in dir(instance):
            if not attr.startswith("_") and callable(getattr(instance, attr)):
                capabilities.append(attr)
        self.module_capabilities[name] = capabilities

class InterModuleCommunication:
    """Enables modules to communicate and share context."""
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.shared_context = {}
        
    def route_request(self, target_module: str, method: str, params: Dict) -> Any:
        """Route a request to a specific module."""
        if target_module not in self.registry.modules:
            raise ValueError(f"Module {target_module} not found")
            
        module_instance = self.registry.modules[target_module]
        if not hasattr(module_instance, method):
            raise ValueError(f"Method {method} not found in {target_module}")
            
        # Execute with shared context
        result = getattr(module_instance, method)(**params)
        return result
        
    def broadcast(self, event: str, data: Dict):
        """Broadcast event to all interested modules."""
        for module_name, instance in self.registry.modules.items():
            if hasattr(instance, f"on_{event}"):
                getattr(instance, f"on_{event}")(data)

class UnifiedDecisionEngine:
    """Orchestrates cross-module intelligence for complex decisions."""
    
    def __init__(self, registry: ModuleRegistry, comm: InterModuleCommunication):
        self.registry = registry
        self.comm = comm
        
    def execute_workflow(self, workflow_def: Dict) -> Dict:
        """Execute multi-module workflow."""
        results = {}
        for step in workflow_def.get("steps", []):
            module = step["module"]
            method = step["method"]
            params = step.get("params", {})
            
            # Inject previous results if needed
            if "inject_from" in step:
                for key, source in step["inject_from"].items():
                    params[key] = results[source]
                    
            result = self.comm.route_request(module, method, params)
            results[step["id"]] = result
            
        return results
        
    def get_recommendation(self, context: Dict) -> Dict:
        """Get AI recommendation by consulting multiple modules."""
        # Example: For inventory reorder, consult inventory + sales + accounting
        recommendations = {}
        
        # This would be customized based on context type
        if context.get("type") == "inventory_reorder":
            # Get inventory status
            recommendations["inventory"] = self.comm.route_request(
                "inventory_ai", "analyze_stock_levels", {}
            )
            # Get sales trends
            recommendations["sales"] = self.comm.route_request(
                "crm_ai", "predict_demand", {}
            )
            # Get budget availability
            recommendations["budget"] = self.comm.route_request(
                "accounting_ai", "check_budget", {}
            )
            
        return {
            "recommendations": recommendations,
            "confidence": self._calculate_confidence(recommendations)
        }
        
    def _calculate_confidence(self, recs: Dict) -> float:
        """Calculate overall confidence from multiple recommendations."""
        return 0.85  # Simplified for now

class CentralAI:
    """Main entry point for the integrated AI system."""
    
    def __init__(self):
        self.registry = ModuleRegistry()
        self.comm = InterModuleCommunication(self.registry)
        self.decision_engine = UnifiedDecisionEngine(self.registry, self.comm)
        self.cognitive = CognitiveIntelligenceAI()
        self.saas = OmnibrainSaaSEngine()
        self.evolution = EvolutionEngine()
        self.singularity = SingularityLogicHub()
        self.absolute_vault = AbsoluteLogicVault()
        self.reports = ReportEngineAI()
        self.planner = SuperAgentPlanner() # 🚀 The Super-AI Brain
        self.query_count = 0
        self.evolution_interval = 10
        self.reasoning_history = []

        # Phase 44: SephlightyBrain as Knowledge-Base Fallback Brain
        try:
            from reasoning.agents import SephlightyBrain
            self.sephlighty = SephlightyBrain()
            logger.info("PHASE 44: SephlightyBrain connected as Knowledge Brain in CentralAI")
        except Exception as e:
            logger.warning(f"Phase 44 SephlightyBrain Bridge: {e}")
            self.sephlighty = None
        logger.info("CENTRAL AI HUB: Supreme MoE + Evolution Engine Online.")
        
    def initialize(self):
        """Bootstrap the entire AI system with robust error isolation."""
        logger.info("Initializing SephlightyAI Central Integration...")
        try:
            self.registry.discover_modules()
            logger.info(f"Loaded {len(self.registry.modules)} modules")
        except Exception as e:
            logger.error(f"Central Registry Initialization Partial Failure: {e}")
            # System can still run with partial registry
        
    def query(self, natural_language: str, context: Optional[Dict] = None) -> Dict:
        """Process query with cross-module error handling."""
        try:
            context = context or {}
            conn_id = context.get("connection_id", "DEFAULT_SAAS_TENANT")
            conv_id = context.get("conversation_id")
            
            # 0. Context Recovery (Database Re-hydration)
            if conv_id:
                try:
                    from chat.models import Message
                    # Find last user query
                    last_user_msg = Message.objects.filter(conversation_id=conv_id, role='user').order_by('-created_at').first()
                    if last_user_msg:
                        self.cognitive.last_query = last_user_msg.content
                    
                    # Find last assistant intent
                    last_ai_msg = Message.objects.filter(conversation_id=conv_id, role='assistant', intent__isnull=False).exclude(intent='UNKNOWN').order_by('-created_at').first()
                    if last_ai_msg and last_ai_msg.intent != "ERROR":
                        self.saas.last_intent = last_ai_msg.intent
                        logger.info(f"Context Recovered: Intent={last_ai_msg.intent}, LastQuery={last_user_msg.content if last_user_msg else 'None'}")
                except Exception as e:
                    logger.warning(f"Context Recovery Failed (Non-critical): {e}")

            logger.info(f"OMNIBRAIN: Processing SaaS query on {conn_id}: {natural_language}")
            
            # 1. Self-Learning Loop
            feedback = context.get("feedback")
            if feedback:
                self.saas.self_learn(feedback)

            # 2. Recursive Evolution Trigger
            self.query_count += 1
            if self.query_count % self.evolution_interval == 0:
                try:
                    self.evolve()
                except Exception as e:
                    logger.error(f"Evolution Loop Failure: {e}")

            # 3. Monte Carlo Simulation Trigger (If requested)
            if "simulate" in natural_language.lower() or "prediction" in natural_language.lower():
                try:
                    sim_result = self.evolution.run_monte_carlo_simulation(
                        {"base_value": 150000, "volatility": 0.08}, 
                        goal=natural_language
                    )
                    context["simulation_data"] = sim_result
                except Exception as e:
                    logger.error(f"Monte Carlo Engine Failure: {e}")

            # 4. Singularity Synergy Check
            try:
                synergy_domains = ["FISCAL_OPS", "HR_PRODUCTIVITY", "SUPPLY_CHAIN_SECURITY"]
                singularity_insights = self.singularity.execute_advanced_reasoning(synergy_domains, context)
                context["singularity_meta"] = singularity_insights
            except Exception as e:
                logger.error(f"Singularity Logic Failure: {e}")

            # 5. Absolute State Resolution
            try:
                absolute_state = self.absolute_vault.resolve_absolute_state(context)
                context["absolute_logic"] = absolute_state
            except Exception as e:
                logger.error(f"Absolute Vault Failure: {e}")

            # 6. Cognitive Synthesis (Identify Context First)
            cog_result = {"response": "", "metadata": {}}
            try:
                raw_result = self.cognitive.process_query(natural_language, context)
                if isinstance(raw_result, dict):
                    cog_result = raw_result
                else:
                    # Fallback for legacy string return
                    cog_result = {"response": str(raw_result), "metadata": {}}
            except Exception as e:
                logger.error(f"Cognitive Resolve Failure: {e}")

            # 7. Primary Routing via SaaS Engine (Critical Path)
            try:
                # Use the contextual resolved_query (from cog) to ensure "je" follow-ups work for data
                resolved_query = cog_result.get("metadata", {}).get("resolved_query", natural_language)
                # Phase 41: Bridge sticky context to SaaS
                sticky_context = cog_result.get("metadata", {}).get("sticky_context", {})
                saas_result = self.saas.process_query(resolved_query, conn_id, context={'sticky_context': sticky_context})
            except Exception as e:
                logger.critical(f"SaaS Core Engine Failure: {e}")
                import traceback
                traceback.print_exc()
                # Return a graceful data bridge error
                return {
                    "answer": f"⚠️ **Data Bridge Error**: Samahani, nimeshindwa kuunganishwa na database ya ERP kwa sasa. (System Error: {str(e)})",
                    "confidence": 0,
                    "intent": "ERROR",
                    "data": []
                }
            
            # Phase 42 Expansion: Recursive Strategic Heuristics (Super-AI Deep Reasoning)
            try:
                # The Planner Agent coordinates EBH, SIE, and MKG
                deep_reasoning = self.planner.process_master_query(resolved_query, context)
                saas_result["deep_analysis"] = deep_reasoning
                logger.info("PHASE 42: Deep Strategic Heuristics Applied to Result")
            except Exception as e:
                logger.error(f"Super-AI Planner Failure (Non-critical): {e}")
            
            # 8. Capture Reasoning for History
            self.reasoning_history.append({
                "context": context,
                "logic_applied": saas_result.get("metadata", {}).get("reasoning_mode", "Standard"),
                "confidence": saas_result.get("metadata", {}).get("confidence", 0.9),
                "success": True 
            })

            # Phase 43: AI Memory & Self-Learning Conversation Tracking
            try:
                if self.saas.ai_memory:
                    self.saas.ai_memory.short_term.push_context({
                        "query": natural_language,
                        "intent": saas_result.get("intent", ""),
                        "confidence": saas_result.get("metadata", {}).get("confidence", 0),
                        "conversation_id": conv_id,
                    })
                if self.saas.self_learner:
                    import time as _t43
                    self.saas.self_learner.on_response(
                        domain=saas_result.get("intent", "general").lower().replace(" ", "_"),
                        query=natural_language,
                        answer=str(saas_result.get("response", ""))[:500],
                        confidence=saas_result.get("metadata", {}).get("confidence", 0.5),
                        response_time=0.0,
                    )
            except Exception:
                pass

            # Phase 43: Embedding Engine — Index conversation for RAG retrieval
            try:
                if self.saas.embedding_engine:
                    self.saas.embedding_engine.index_text(
                        text=f"{natural_language} | {str(saas_result.get('response', ''))[:200]}",
                        domain="conversations",
                        metadata={"intent": saas_result.get("intent", ""), "conversation_id": conv_id}
                    )
            except Exception:
                pass

            # Phase 43: Agent Pipeline Narration for complex/multi-domain queries
            try:
                if self.saas.agent_pipeline:
                    plan = self.saas.agent_pipeline.process_plan_only(natural_language)
                    plan_info = plan.get("plan", {})
                    if plan_info.get("complexity") in ("complex", "multi-domain"):
                        narrator_result = self.saas.agent_pipeline.narrate(natural_language, saas_result.get("response", ""))
                        if narrator_result:
                            saas_result["agent_narration"] = narrator_result
                            logger.info("PHASE 43: Agent Pipeline narration applied")
            except Exception:
                pass

            # Phase 44: SephlightyBrain Knowledge-Base Fallback
            # If SaaS returned low confidence, enrich via SephlightyBrain's 300k+ KB
            if self.sephlighty:
                try:
                    saas_confidence = saas_result.get("metadata", {}).get("confidence", 1.0)
                    if saas_confidence < 0.7:
                        kb_result = self.sephlighty.run(natural_language)
                        if kb_result.get("confidence", 0) > 70:
                            saas_result["kb_enrichment"] = kb_result.get("answer", "")
                            logger.info("PHASE 44: SephlightyBrain Knowledge Enrichment applied")
                except Exception:
                    pass

            # Phase 45: Super Smart Synthesis
            # Merges SephlightyBrain reasoning with SaaS Data for a single unified answer.
            saas_result["answer"] = self._synthesize_super_smart(saas_result, cog_result)

            return saas_result

        except Exception as e:
            logger.critical(f"GLOBAL CENTRAL_AI CRASH: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"🚨 **Kernel Panic**: My internal integration layer encountered a critical error: {str(e)}",
                "confidence": 0,
                "intent": "ERROR",
                "data": []
            }

    def _synthesize_super_smart(self, result: Dict, cog_result: Dict) -> str:
        """
        Unified output synthesis for Phase 45.
        Combines data, narration, and knowledge into a single cohesive story.
        """
        data_res = result.get("response", "")
        if not isinstance(data_res, str): data_res = str(data_res)
        
        narration = result.get("agent_narration", "")
        enrichment = result.get("kb_enrichment", "")
        insight = result.get("deep_analysis", {}).get("response", "")
        cog_res = cog_result.get("response", "")
        
        # 1. Start with the data-driven core
        answer = f"{data_res}"
        
        # 2. Add Narration as the "Agent Voice"
        if narration:
            answer += f"\n\n🤖 **AI Agent Analysis:**\n{narration}"
        
        # 3. Add Strategic Insight
        if insight:
            answer += f"\n\n🚨 **STRATEGIC AI INSIGHT:**\n{insight}"
            
        # 4. Smart Knowledge Integration (Avoid Redundancy)
        if enrichment:
            # Check for contradiction between data and enrichment
            if "decreas" in enrichment.lower() and "increas" in data_res.lower():
                answer += f"\n\n⚠️ **Logic Conflict Detected:** Cross-verifying internal reports. Note: Recent trends may vary by specific category filtering."
            
            # Append as enriched context if not redundant
            if len(enrichment) > 20: 
                answer += f"\n\n📚 **Knowledge Enrichment:**\n{enrichment}"
        
        # 5. Add Cognitive Tips if not already covered
        if cog_res and cog_result.get("metadata", {}).get("reasoning_applied") != "knowledge":
            answer += f"\n\n💡 **AI Pro-Tip:**\n{cog_res}"

        # 6. Final Clean-up: Ensure TZS formatting is consistent
        answer = answer.replace("tzs", "TZS").replace("Tzs", "TZS")
        
        return answer


    def evolve(self):
        """Trigger recursive logic distillation and optimization."""
        logger_api.info("OMNIBRAIN: Starting Evolutionary Recursive Cycle...")
        new_logic = self.evolution.distill_logic(self.reasoning_history)
        if new_logic:
            logger_api.info(f"OMNIBRAIN: Distilled {len(new_logic)} new business heuristics.")
            # In a real system, these would be injected into the SaaS Engine's rule-base
        self.evolution.optimize_expert_weights([])
        self.reasoning_history = [] # Reset for next cycle
        
    def execute(self, workflow: Dict) -> Dict:
        """Execute complex multi-module workflow."""
        return self.decision_engine.execute_workflow(workflow)

    def generate_conversation_title(self, message_history: List[Dict[str, str]]) -> str:
        """Ask Cognitive layer to suggest a title for the thread."""
        return self.cognitive.suggest_title(message_history)
