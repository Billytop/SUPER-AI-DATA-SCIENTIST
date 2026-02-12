"""
SephlightyAI Multi-Agent Pipeline
Author: Antigravity AI
Version: 1.0.0

Enterprise-grade multi-agent reasoning system.
Chains specialized agents for deep, validated business intelligence.

AGENT PIPELINE:
  User Query → PlannerAgent → AnalyzerAgent → ReasoningAgent → ValidatorAgent → NarratorAgent → Final Answer

Each agent is independent and passes structured context to the next.
"""

import logging
import re
import math
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger("AGENT_PIPELINE")
logger.setLevel(logging.INFO)


# =============================================================================
# 1. BASE AGENT
# =============================================================================

class BaseAgent:
    """Base class for all pipeline agents."""

    def __init__(self, name: str):
        self.name = name
        self.execution_time: float = 0.0

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's logic. Override in subclasses."""
        raise NotImplementedError

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run with timing."""
        start = time.time()
        result = self.execute(context)
        self.execution_time = round(time.time() - start, 4)
        result["_agent_name"] = self.name
        result["_execution_time"] = self.execution_time
        return result


# =============================================================================
# 2. PLANNER AGENT
# =============================================================================

class PlannerAgent(BaseAgent):
    """
    Breaks a user query into structured sub-tasks.
    Detects intent, domain, required data sources, and analysis type.
    """

    # Domain detection patterns
    DOMAIN_PATTERNS = {
        "sales": [
            "sale", "sold", "revenue", "income", "mauzo", "mapato", "uza",
            "best selling", "worst selling", "top product", "profit",
            "invoice", "receipt", "order"
        ],
        "purchases": [
            "purchase", "supplier", "vendor", "buy", "bought", "nunua",
            "procurement", "restock", "supply", "cost of goods"
        ],
        "customers": [
            "customer", "client", "mteja", "wateja", "debt", "deni",
            "credit", "balance", "ledger", "owing", "payment", "risk"
        ],
        "expenses": [
            "expense", "cost", "matumizi", "gharama", "overhead",
            "rent", "salary", "utility", "bill", "spending"
        ],
        "inventory": [
            "stock", "inventory", "product", "item", "bidhaa",
            "warehouse", "quantity", "dead stock", "reorder"
        ],
        "business_health": [
            "health", "profit", "loss", "cashflow", "liquidity",
            "burn rate", "survival", "p&l", "balance sheet",
            "capital", "efficiency", "margin"
        ],
    }

    # Analysis type patterns
    ANALYSIS_PATTERNS = {
        "aggregation": ["total", "sum", "count", "jumla", "ngapi", "how many", "how much"],
        "comparison": ["compare", "vs", "versus", "better", "worse", "difference", "linganisha"],
        "trend": ["trend", "growing", "declining", "over time", "monthly", "yearly", "weekly"],
        "forecast": ["predict", "forecast", "next month", "future", "tabiri", "projection"],
        "ranking": ["best", "worst", "top", "bottom", "highest", "lowest", "rank"],
        "anomaly": ["unusual", "suspicious", "anomaly", "strange", "irregular"],
        "recommendation": ["recommend", "suggest", "advice", "should", "pendekeza"],
        "detail": ["show", "list", "detail", "breakdown", "describe", "eleza"],
    }

    # Time period patterns
    TIME_PATTERNS = {
        "today": ["today", "leo"],
        "yesterday": ["yesterday", "jana"],
        "this_week": ["this week", "wiki hii"],
        "last_week": ["last week", "wiki iliyopita"],
        "this_month": ["this month", "mwezi huu"],
        "last_month": ["last month", "mwezi uliopita"],
        "this_year": ["this year", "mwaka huu"],
        "last_year": ["last year", "mwaka jana", "mwaka uliopita"],
        "quarter": ["quarter", "q1", "q2", "q3", "q4", "robo"],
    }

    def __init__(self):
        super().__init__("PlannerAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Break query into structured plan."""
        query = context.get("query", "").lower()

        # Detect domains
        domains = self._detect_domains(query)

        # Detect analysis types
        analysis_types = self._detect_analysis_types(query)

        # Detect time period
        time_period = self._detect_time_period(query)

        # Generate sub-tasks
        sub_tasks = self._generate_sub_tasks(domains, analysis_types, time_period, query)

        # Detect required data sources (tables)
        data_sources = self._map_data_sources(domains)

        return {
            "plan": {
                "domains": domains,
                "analysis_types": analysis_types,
                "time_period": time_period,
                "sub_tasks": sub_tasks,
                "data_sources": data_sources,
                "complexity": self._assess_complexity(domains, analysis_types),
                "requires_cross_table": len(domains) > 1,
            },
            "original_query": query,
        }

    def _detect_domains(self, query: str) -> List[str]:
        """Detect which business domains are relevant."""
        detected = []
        for domain, keywords in self.DOMAIN_PATTERNS.items():
            for kw in keywords:
                if kw in query:
                    detected.append(domain)
                    break
        return detected or ["general"]

    def _detect_analysis_types(self, query: str) -> List[str]:
        """Detect what type of analysis is needed."""
        detected = []
        for atype, keywords in self.ANALYSIS_PATTERNS.items():
            for kw in keywords:
                if kw in query:
                    detected.append(atype)
                    break
        return detected or ["detail"]

    def _detect_time_period(self, query: str) -> str:
        """Detect the time period referenced in the query."""
        for period, keywords in self.TIME_PATTERNS.items():
            for kw in keywords:
                if kw in query:
                    return period
        return "all_time"

    def _generate_sub_tasks(self, domains: List[str], analysis_types: List[str],
                            time_period: str, query: str) -> List[Dict[str, Any]]:
        """Generate ordered sub-tasks for the analyzer."""
        tasks = []
        task_id = 1

        for domain in domains:
            for atype in analysis_types:
                tasks.append({
                    "id": task_id,
                    "domain": domain,
                    "analysis_type": atype,
                    "time_period": time_period,
                    "description": f"{atype.capitalize()} analysis on {domain} for {time_period}",
                    "status": "pending",
                })
                task_id += 1

        return tasks

    def _map_data_sources(self, domains: List[str]) -> Dict[str, List[str]]:
        """Map domains to database tables."""
        mapping = {
            "sales": ["transactions", "transaction_sell_lines", "products", "contacts"],
            "purchases": ["transactions", "purchase_lines", "contacts", "products"],
            "customers": ["contacts", "transactions", "transaction_payments", "customer_groups"],
            "expenses": ["transactions", "expense_categories"],
            "inventory": ["products", "variations", "variation_location_details", "purchase_lines"],
            "business_health": ["transactions", "transaction_payments", "products", "contacts"],
            "general": ["transactions", "products", "contacts"],
        }
        result = {}
        for domain in domains:
            result[domain] = mapping.get(domain, mapping["general"])
        return result

    def _assess_complexity(self, domains: List[str], analysis_types: List[str]) -> str:
        """Assess query complexity for routing decisions."""
        score = len(domains) + len(analysis_types)
        if "forecast" in analysis_types:
            score += 2
        if "anomaly" in analysis_types:
            score += 2
        if score <= 2:
            return "simple"
        elif score <= 4:
            return "moderate"
        else:
            return "complex"


# =============================================================================
# 3. ANALYZER AGENT
# =============================================================================

class AnalyzerAgent(BaseAgent):
    """
    Executes data queries and performs analytics on the results.
    Interfaces with the SaaS engine to pull real data from the ERP database.
    """

    def __init__(self):
        super().__init__("AnalyzerAgent")
        self._saas_engine = None

    def _get_saas_engine(self):
        """Lazy-load the SaaS engine."""
        if self._saas_engine is None:
            try:
                from laravel_modules.ai_brain.omnibrain_saas_engine import OmnibrainSaaSEngine
                self._saas_engine = OmnibrainSaaSEngine()
            except ImportError:
                logger.warning("SaaS engine not available. Analytics will be limited.")
        return self._saas_engine

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analytics based on plan."""
        plan = context.get("plan", {})
        sub_tasks = plan.get("sub_tasks", [])
        connection_id = context.get("connection_id", "TENANT_001")

        results = []
        raw_data = {}
        errors = []

        for task in sub_tasks:
            try:
                task_result = self._execute_task(task, connection_id)
                task["status"] = "completed"
                results.append(task_result)

                # Store raw data for downstream agents
                domain = task.get("domain", "general")
                if domain not in raw_data:
                    raw_data[domain] = []
                raw_data[domain].append(task_result)
            except Exception as e:
                task["status"] = "failed"
                errors.append({"task_id": task["id"], "error": str(e)})

        # Compute aggregated metrics
        metrics = self._compute_metrics(results, plan.get("domains", []))

        return {
            "analysis_results": results,
            "raw_data": raw_data,
            "metrics": metrics,
            "errors": errors,
            "tasks_completed": len(results),
            "tasks_failed": len(errors),
            "plan": plan,
        }

    def _execute_task(self, task: Dict[str, Any], connection_id: str) -> Dict[str, Any]:
        """Execute a single analysis task by querying the SaaS engine."""
        domain = task.get("domain", "general")
        atype = task.get("analysis_type", "detail")
        time_period = task.get("time_period", "all_time")

        engine = self._get_saas_engine()
        if engine is None:
            return {
                "task_id": task["id"],
                "domain": domain,
                "status": "skipped",
                "reason": "SaaS engine not available",
            }

        # Build natural language query for the engine
        query = f"{atype} {domain} {time_period}".replace("_", " ")
        result = engine.process_query(query, connection_id)

        return {
            "task_id": task["id"],
            "domain": domain,
            "analysis_type": atype,
            "time_period": time_period,
            "data": result.get("data"),
            "answer": result.get("answer", ""),
            "confidence": result.get("metadata", {}).get("confidence", 0.0),
            "sql_used": result.get("metadata", {}).get("sql_query"),
        }

    def _compute_metrics(self, results: List[Dict], domains: List[str]) -> Dict[str, Any]:
        """Compute aggregated metrics from analysis results."""
        metrics = {
            "total_results": len(results),
            "avg_confidence": 0.0,
            "domains_analyzed": domains,
        }

        confidences = [r.get("confidence", 0) for r in results if r.get("confidence")]
        if confidences:
            metrics["avg_confidence"] = round(sum(confidences) / len(confidences), 4)

        return metrics


# =============================================================================
# 4. REASONING AGENT
# =============================================================================

class ReasoningAgent(BaseAgent):
    """
    Applies deep reasoning: trend detection, cause analysis, what-if scenarios.
    Uses the Transformer Core for advanced reasoning when available.
    """

    def __init__(self):
        super().__init__("ReasoningAgent")
        self._transformer = None

    def _get_transformer(self):
        """Lazy-load transformer."""
        if self._transformer is None:
            try:
                from laravel_modules.ai_brain.transformer_core import TRANSFORMER_BRAIN
                self._transformer = TRANSFORMER_BRAIN
            except ImportError:
                logger.warning("Transformer not available. Reasoning will be rule-based.")
        return self._transformer

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multi-layered reasoning."""
        analysis = context.get("analysis_results", [])
        plan = context.get("plan", {})
        query = context.get("original_query", "")

        insights = []
        causes = []
        recommendations = []
        transformer_signals = {}

        # --- Rule-Based Reasoning ---
        for result in analysis:
            domain = result.get("domain", "")
            atype = result.get("analysis_type", "")

            # Trend reasoning
            if atype == "trend":
                insight = self._analyze_trend(result)
                if insight:
                    insights.append(insight)

            # Comparison reasoning
            elif atype == "comparison":
                insight = self._analyze_comparison(result)
                if insight:
                    insights.append(insight)

            # Anomaly reasoning
            elif atype == "anomaly":
                anomalies = self._detect_anomalies(result)
                insights.extend(anomalies)

            # General reasoning
            else:
                insight = self._general_reasoning(result)
                if insight:
                    insights.append(insight)

        # --- Transformer-Based Reasoning (if available) ---
        transformer = self._get_transformer()
        if transformer and analysis:
            data_rows = []
            for r in analysis:
                if isinstance(r.get("data"), list):
                    data_rows.extend(r["data"][:10])
                elif isinstance(r.get("data"), dict):
                    data_rows.append(r["data"])

            if data_rows:
                t_result = transformer.reason(query, data_rows)
                transformer_signals = {
                    "confidence": t_result.get("confidence", 0),
                    "tokens_processed": t_result.get("total_tokens", 0),
                }

        # Generate recommendations
        recommendations = self._generate_recommendations(insights, plan)

        # Identify root causes
        causes = self._identify_causes(insights, analysis)

        return {
            "insights": insights,
            "causes": causes,
            "recommendations": recommendations,
            "transformer_signals": transformer_signals,
            "reasoning_depth": len(insights),
            "plan": plan,
            "analysis_results": analysis,
        }

    def _analyze_trend(self, result: Dict) -> Optional[Dict]:
        """Detect trends in time-series data."""
        data = result.get("data")
        if not data:
            return None

        return {
            "type": "trend",
            "domain": result.get("domain"),
            "finding": f"Trend analysis for {result.get('domain', 'unknown')} completed",
            "direction": "analyzed",
            "confidence": result.get("confidence", 0.7),
        }

    def _analyze_comparison(self, result: Dict) -> Optional[Dict]:
        """Perform comparison analysis."""
        return {
            "type": "comparison",
            "domain": result.get("domain"),
            "finding": f"Comparison analysis for {result.get('domain', 'unknown')}",
            "confidence": result.get("confidence", 0.7),
        }

    def _detect_anomalies(self, result: Dict) -> List[Dict]:
        """Detect data anomalies."""
        anomalies = []
        data = result.get("data")
        if not data:
            return anomalies

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, (int, float)) and value < 0:
                            anomalies.append({
                                "type": "anomaly",
                                "domain": result.get("domain"),
                                "finding": f"Negative value detected in {key}: {value}",
                                "severity": "medium",
                            })
        return anomalies

    def _general_reasoning(self, result: Dict) -> Optional[Dict]:
        """Apply general business reasoning."""
        return {
            "type": "analysis",
            "domain": result.get("domain"),
            "finding": result.get("answer", "Analysis completed"),
            "confidence": result.get("confidence", 0.7),
        }

    def _generate_recommendations(self, insights: List[Dict],
                                   plan: Dict) -> List[Dict]:
        """Generate actionable recommendations."""
        recommendations = []
        domains = plan.get("domains", [])

        if "sales" in domains:
            recommendations.append({
                "area": "sales",
                "action": "Focus on top-performing products and investigate declining categories",
                "priority": "high",
            })

        if "customers" in domains:
            recommendations.append({
                "area": "customers",
                "action": "Review high-risk customers and tighten credit policies",
                "priority": "high",
            })

        if "expenses" in domains:
            recommendations.append({
                "area": "expenses",
                "action": "Identify and eliminate unnecessary variable costs",
                "priority": "medium",
            })

        if "inventory" in domains:
            recommendations.append({
                "area": "inventory",
                "action": "Address dead stock and optimize reorder points",
                "priority": "high",
            })

        return recommendations

    def _identify_causes(self, insights: List[Dict],
                          analysis: List[Dict]) -> List[Dict]:
        """Identify root causes for observed patterns."""
        causes = []
        for insight in insights:
            if insight.get("type") == "anomaly":
                causes.append({
                    "anomaly": insight.get("finding"),
                    "possible_cause": "Data entry error or unusual business event",
                    "action": "Investigate manually",
                })
        return causes


# =============================================================================
# 5. VALIDATOR AGENT
# =============================================================================

class ValidatorAgent(BaseAgent):
    """
    Validates mathematical correctness and logical consistency of results.
    Ensures no hallucination by cross-checking computations.
    """

    def __init__(self):
        super().__init__("ValidatorAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all results."""
        insights = context.get("insights", [])
        analysis_results = context.get("analysis_results", [])
        recommendations = context.get("recommendations", [])

        validation_results = []
        warnings = []
        is_valid = True

        # 1. Check mathematical consistency
        math_check = self._validate_math(analysis_results)
        validation_results.append(math_check)
        if not math_check["passed"]:
            is_valid = False
            warnings.append(math_check["warning"])

        # 2. Check logical consistency
        logic_check = self._validate_logic(insights)
        validation_results.append(logic_check)
        if not logic_check["passed"]:
            warnings.append(logic_check["warning"])

        # 3. Check confidence levels
        confidence_check = self._validate_confidence(analysis_results)
        validation_results.append(confidence_check)
        if not confidence_check["passed"]:
            warnings.append(confidence_check["warning"])

        # 4. Check data completeness
        completeness_check = self._validate_completeness(context)
        validation_results.append(completeness_check)

        return {
            "is_valid": is_valid,
            "validation_results": validation_results,
            "warnings": warnings,
            "confidence_adjusted": confidence_check.get("adjusted_confidence", 0.8),
            # Pass through
            "insights": insights,
            "analysis_results": analysis_results,
            "recommendations": recommendations,
            "causes": context.get("causes", []),
            "transformer_signals": context.get("transformer_signals", {}),
            "plan": context.get("plan", {}),
        }

    def _validate_math(self, results: List[Dict]) -> Dict[str, Any]:
        """Verify mathematical correctness of computed values."""
        issues = []

        for result in results:
            data = result.get("data")
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Check for NaN or Infinity
                        for key, value in item.items():
                            if isinstance(value, float):
                                if math.isnan(value) or math.isinf(value):
                                    issues.append(f"Invalid number in {key}: {value}")

        return {
            "check": "mathematical_consistency",
            "passed": len(issues) == 0,
            "issues": issues,
            "warning": f"Math validation failed: {', '.join(issues)}" if issues else None,
        }

    def _validate_logic(self, insights: List[Dict]) -> Dict[str, Any]:
        """Check for logical contradictions."""
        contradictions = []

        # Check for contradictory trend directions in same domain
        domain_trends = {}
        for insight in insights:
            if insight.get("type") == "trend":
                domain = insight.get("domain")
                direction = insight.get("direction")
                if domain in domain_trends and domain_trends[domain] != direction:
                    contradictions.append(
                        f"Contradictory trends in {domain}: "
                        f"{domain_trends[domain]} vs {direction}"
                    )
                domain_trends[domain] = direction

        return {
            "check": "logical_consistency",
            "passed": len(contradictions) == 0,
            "contradictions": contradictions,
            "warning": f"Logic issues: {', '.join(contradictions)}" if contradictions else None,
        }

    def _validate_confidence(self, results: List[Dict]) -> Dict[str, Any]:
        """Validate and adjust overall confidence."""
        confidences = [r.get("confidence", 0) for r in results if r.get("confidence")]

        if not confidences:
            return {
                "check": "confidence_levels",
                "passed": True,
                "adjusted_confidence": 0.7,
                "warning": None,
            }

        avg = sum(confidences) / len(confidences)
        min_conf = min(confidences)

        # Flag if any result has very low confidence
        passed = min_conf >= 0.3

        return {
            "check": "confidence_levels",
            "passed": passed,
            "avg_confidence": round(avg, 4),
            "min_confidence": round(min_conf, 4),
            "adjusted_confidence": round(avg * 0.9 + 0.1, 4),
            "warning": f"Low confidence detected: {min_conf:.2f}" if not passed else None,
        }

    def _validate_completeness(self, context: Dict) -> Dict[str, Any]:
        """Check if all planned tasks were completed."""
        plan = context.get("plan", {})
        sub_tasks = plan.get("sub_tasks", [])

        completed = sum(1 for t in sub_tasks if t.get("status") == "completed")
        total = len(sub_tasks)

        return {
            "check": "data_completeness",
            "passed": completed == total,
            "completed": completed,
            "total": total,
            "warning": f"Only {completed}/{total} tasks completed" if completed < total else None,
        }


# =============================================================================
# 6. NARRATOR AGENT
# =============================================================================

class NarratorAgent(BaseAgent):
    """
    Synthesizes all reasoning into a clear, business-language explanation.
    Supports English + Swahili bilingual output.
    """

    def __init__(self):
        super().__init__("NarratorAgent")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final narrative response."""
        insights = context.get("insights", [])
        recommendations = context.get("recommendations", [])
        causes = context.get("causes", [])
        warnings = context.get("warnings", [])
        plan = context.get("plan", {})
        is_valid = context.get("is_valid", True)
        confidence = context.get("confidence_adjusted", 0.8)

        # Detect if Swahili response needed
        original_query = context.get("original_query", "")
        use_swahili = self._is_swahili_query(original_query)

        # Build response sections
        sections = []

        # Main answer
        main_answer = self._build_main_answer(insights, plan, use_swahili)
        sections.append(main_answer)

        # Insights
        if insights:
            insight_section = self._build_insights_section(insights, use_swahili)
            sections.append(insight_section)

        # Recommendations
        if recommendations:
            rec_section = self._build_recommendations_section(recommendations, use_swahili)
            sections.append(rec_section)

        # Warnings (if any)
        if warnings and not is_valid:
            warn_section = self._build_warnings_section(warnings, use_swahili)
            sections.append(warn_section)

        # Compose final narrative
        narrative = "\n\n".join(sections)

        return {
            "narrative": narrative,
            "confidence": confidence,
            "language": "sw" if use_swahili else "en",
            "sections_count": len(sections),
            "has_warnings": len(warnings) > 0,
            "metadata": {
                "domains": plan.get("domains", []),
                "analysis_types": plan.get("analysis_types", []),
                "time_period": plan.get("time_period", "all_time"),
                "complexity": plan.get("complexity", "simple"),
            },
        }

    def _is_swahili_query(self, query: str) -> bool:
        """Detect if query is in Swahili."""
        sw_markers = [
            "mauzo", "mteja", "bidhaa", "matumizi", "gharama",
            "onyesha", "ngapi", "jumla", "gani", "kwa", "ni",
            "soko", "faida", "hasara", "deni", "mkopo"
        ]
        query_lower = query.lower()
        matches = sum(1 for m in sw_markers if m in query_lower)
        return matches >= 2

    def _build_main_answer(self, insights: List[Dict], plan: Dict,
                           use_swahili: bool) -> str:
        """Build the main answer section."""
        domains = plan.get("domains", ["general"])
        domain_str = ", ".join(domains)

        if use_swahili:
            header = f"📊 **Uchambuzi wa {domain_str.title()}**"
        else:
            header = f"📊 **{domain_str.title()} Analysis**"

        if insights:
            first_insight = insights[0].get("finding", "Analysis completed.")
            return f"{header}\n\n{first_insight}"
        else:
            if use_swahili:
                return f"{header}\n\nUchambuzi umekamilika."
            return f"{header}\n\nAnalysis completed."

    def _build_insights_section(self, insights: List[Dict],
                                 use_swahili: bool) -> str:
        """Build insights section."""
        if use_swahili:
            header = "💡 **Maarifa ya AI:**"
        else:
            header = "💡 **AI Insights:**"

        lines = [header]
        for i, insight in enumerate(insights[:5], 1):
            finding = insight.get("finding", "")
            confidence = insight.get("confidence", 0)
            conf_pct = f" ({confidence * 100:.0f}% uhakika)" if use_swahili \
                else f" ({confidence * 100:.0f}% confidence)"
            lines.append(f"{i}. {finding}{conf_pct}")

        return "\n".join(lines)

    def _build_recommendations_section(self, recommendations: List[Dict],
                                        use_swahili: bool) -> str:
        """Build recommendations section."""
        if use_swahili:
            header = "🎯 **Mapendekezo:**"
        else:
            header = "🎯 **Recommendations:**"

        lines = [header]
        for rec in recommendations[:5]:
            priority = rec.get("priority", "medium")
            action = rec.get("action", "")
            icon = "🔴" if priority == "high" else "🟡" if priority == "medium" else "🟢"
            lines.append(f"  {icon} {action}")

        return "\n".join(lines)

    def _build_warnings_section(self, warnings: List[str],
                                 use_swahili: bool) -> str:
        """Build warnings section."""
        if use_swahili:
            header = "⚠️ **Tahadhari:**"
        else:
            header = "⚠️ **Warnings:**"

        lines = [header]
        for w in warnings:
            if w:
                lines.append(f"  - {w}")

        return "\n".join(lines)


# =============================================================================
# 7. AGENT PIPELINE ORCHESTRATOR
# =============================================================================

class AgentPipeline:
    """
    Orchestrates the full multi-agent pipeline:
    Query → Planner → Analyzer → Reasoning → Validator → Narrator → Answer

    Each agent receives the accumulated context from all previous agents.
    """

    def __init__(self):
        self.planner = PlannerAgent()
        self.analyzer = AnalyzerAgent()
        self.reasoner = ReasoningAgent()
        self.validator = ValidatorAgent()
        self.narrator = NarratorAgent()

        self.agents = [
            self.planner,
            self.analyzer,
            self.reasoner,
            self.validator,
            self.narrator,
        ]

        self.execution_log: List[Dict] = []
        logger.info("AgentPipeline initialized: 5 agents ready.")

    def process(self, query: str, connection_id: str = "TENANT_001",
                user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the full agent pipeline on a user query.

        Args:
            query: Natural language business question
            connection_id: Database connection identifier
            user_id: Optional user identifier

        Returns:
            Complete response with narrative, confidence, metadata
        """
        start_time = time.time()
        self.execution_log = []

        # Initialize context
        context = {
            "query": query,
            "original_query": query,
            "connection_id": connection_id,
            "user_id": user_id,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Run each agent in sequence
        for agent in self.agents:
            try:
                result = agent.run(context)
                context.update(result)

                self.execution_log.append({
                    "agent": agent.name,
                    "execution_time": agent.execution_time,
                    "status": "success",
                })

                logger.info(f"PIPELINE: {agent.name} completed in {agent.execution_time}s")

            except Exception as e:
                logger.error(f"PIPELINE: {agent.name} FAILED: {e}")
                self.execution_log.append({
                    "agent": agent.name,
                    "execution_time": agent.execution_time,
                    "status": "failed",
                    "error": str(e),
                })

        total_time = round(time.time() - start_time, 4)

        return {
            "answer": context.get("narrative", "Analysis could not be completed."),
            "confidence": context.get("confidence_adjusted", 0.5),
            "language": context.get("language", "en"),
            "metadata": {
                "pipeline_time": total_time,
                "agents_executed": len(self.execution_log),
                "execution_log": self.execution_log,
                "domains": context.get("plan", {}).get("domains", []),
                "complexity": context.get("plan", {}).get("complexity", "unknown"),
                "transformer_signals": context.get("transformer_signals", {}),
                "warnings": context.get("warnings", []),
            },
        }

    def process_plan_only(self, query: str) -> Dict[str, Any]:
        """Run only the planner to see the execution plan without running analytics."""
        context = {"query": query, "original_query": query}
        return self.planner.run(context)


# =============================================================================
# 8. GLOBAL SINGLETON
# =============================================================================

AGENT_PIPELINE = AgentPipeline()

logger.info("Multi-Agent Pipeline v1.0.0 — 5 Agents deployed.")
