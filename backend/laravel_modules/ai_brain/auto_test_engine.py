"""
SephlightyAI Auto-Test Engine
Author: Antigravity AI
Version: 1.0.0

Automated testing framework for all AI brain components.
Runs unit tests, integration tests, and regression tests automatically.
Self-validates after every upgrade cycle.

TEST CATEGORIES:
  - Unit: Individual component correctness
  - Integration: Cross-module communication
  - Regression: Previously fixed issues don't recur
  - Smoke: Quick system-wide health check
"""

import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger("AUTO_TEST_ENGINE")
logger.setLevel(logging.INFO)


# =============================================================================
# 1. TEST RESULT
# =============================================================================

class TestResult:
    """Individual test result."""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category
        self.passed: bool = False
        self.error: Optional[str] = None
        self.execution_time: float = 0.0
        self.details: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "error": self.error,
            "execution_time": self.execution_time,
            "details": self.details,
        }


# =============================================================================
# 2. TEST SUITE
# =============================================================================

class TestSuite:
    """Collection of related tests."""

    def __init__(self, name: str, category: str = "unit"):
        self.name = name
        self.category = category
        self.tests: List[Callable] = []
        self.results: List[TestResult] = []

    def add_test(self, test_func: Callable) -> None:
        """Add a test function to the suite."""
        self.tests.append(test_func)

    def run(self) -> List[TestResult]:
        """Run all tests in the suite."""
        self.results = []
        for test_func in self.tests:
            result = TestResult(test_func.__name__, self.category)
            start = time.time()
            try:
                details = test_func()
                result.passed = True
                result.details = details if isinstance(details, dict) else {}
            except Exception as e:
                result.passed = False
                result.error = f"{type(e).__name__}: {str(e)}"
            result.execution_time = round(time.time() - start, 4)
            self.results.append(result)

        return self.results


# =============================================================================
# 3. AUTO TEST ENGINE
# =============================================================================

class AutoTestEngine:
    """
    Comprehensive automated testing engine.
    Discovers, runs, and reports on all AI brain tests.
    """

    def __init__(self):
        self.suites: List[TestSuite] = []
        self.last_run_results: List[TestResult] = []
        self.run_history: List[Dict[str, Any]] = []
        self._register_all_suites()
        logger.info("AutoTestEngine initialized.")

    def _register_all_suites(self):
        """Register all built-in test suites."""

        # Suite 1: Transformer Core
        transformer_suite = TestSuite("Transformer Core", "unit")
        transformer_suite.add_test(self._test_transformer_import)
        transformer_suite.add_test(self._test_positional_encoding)
        transformer_suite.add_test(self._test_multihead_attention)
        transformer_suite.add_test(self._test_ffn)
        transformer_suite.add_test(self._test_transformer_block)
        transformer_suite.add_test(self._test_business_embedder)
        transformer_suite.add_test(self._test_transformer_reason)
        self.suites.append(transformer_suite)

        # Suite 2: Agent Pipeline
        agent_suite = TestSuite("Agent Pipeline", "unit")
        agent_suite.add_test(self._test_planner_agent)
        agent_suite.add_test(self._test_validator_agent)
        agent_suite.add_test(self._test_narrator_agent)
        agent_suite.add_test(self._test_pipeline_plan_only)
        self.suites.append(agent_suite)

        # Suite 3: Memory Service
        memory_suite = TestSuite("Memory Service", "unit")
        memory_suite.add_test(self._test_short_term_memory)
        memory_suite.add_test(self._test_memory_search)
        memory_suite.add_test(self._test_context_stack)
        self.suites.append(memory_suite)

        # Suite 4: Embedding Engine
        embedding_suite = TestSuite("Embedding Engine", "unit")
        embedding_suite.add_test(self._test_hash_embedding)
        embedding_suite.add_test(self._test_vector_index)
        embedding_suite.add_test(self._test_cosine_similarity)
        embedding_suite.add_test(self._test_business_embedding)
        self.suites.append(embedding_suite)

        # Suite 5: RAG Engine
        rag_suite = TestSuite("RAG Engine", "unit")
        rag_suite.add_test(self._test_context_builder)
        rag_suite.add_test(self._test_fact_grounder)
        rag_suite.add_test(self._test_rag_augmented_query)
        self.suites.append(rag_suite)

        # Suite 6: Integration
        integration_suite = TestSuite("Integration", "integration")
        integration_suite.add_test(self._test_transformer_to_pipeline)
        integration_suite.add_test(self._test_embedding_to_rag)
        integration_suite.add_test(self._test_memory_to_rag)
        self.suites.append(integration_suite)

        # Suite 7: Smoke Tests
        smoke_suite = TestSuite("Smoke", "smoke")
        smoke_suite.add_test(self._test_all_imports)
        smoke_suite.add_test(self._test_singletons)
        self.suites.append(smoke_suite)

    # ── Transformer Tests ────────────────────────────────────────────────

    def _test_transformer_import(self) -> Dict:
        from laravel_modules.ai_brain.transformer_core import BusinessTransformer
        return {"class": "BusinessTransformer"}

    def _test_positional_encoding(self) -> Dict:
        from laravel_modules.ai_brain.transformer_core import PositionalEncoding
        pe = PositionalEncoding(64, 100)
        enc = pe.encode(0)
        assert len(enc) == 64
        return {"dim": 64}

    def _test_multihead_attention(self) -> Dict:
        from laravel_modules.ai_brain.transformer_core import MultiHeadAttention
        mha = MultiHeadAttention(64, 4)
        x = [[0.1] * 64 for _ in range(3)]
        out = mha.forward(x)
        assert len(out) == 3
        assert len(out[0]) == 64
        return {"seq_len": 3, "d_model": 64}

    def _test_ffn(self) -> Dict:
        from laravel_modules.ai_brain.transformer_core import FeedForwardNetwork
        ffn = FeedForwardNetwork(64)
        x = [0.5] * 64
        out = ffn.forward(x)
        assert len(out) == 64
        return {"dim": 64}

    def _test_transformer_block(self) -> Dict:
        from laravel_modules.ai_brain.transformer_core import TransformerBlock
        block = TransformerBlock(64, 4)
        x = [[0.1] * 64 for _ in range(3)]
        out = block.forward(x)
        assert len(out) == 3
        return {"layers": 1}

    def _test_business_embedder(self) -> Dict:
        from laravel_modules.ai_brain.transformer_core import BusinessTokenEmbedder
        emb = BusinessTokenEmbedder(64)
        vec = emb.embed_token("hello")
        assert len(vec) == 64
        num = emb.embed_number(42000)
        assert len(num) == 64
        return {"dim": 64}

    def _test_transformer_reason(self) -> Dict:
        from laravel_modules.ai_brain.transformer_core import TRANSFORMER_BRAIN
        result = TRANSFORMER_BRAIN.reason("test", [{"a": "1"}])
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
        return {"confidence": result["confidence"]}

    # ── Agent Pipeline Tests ─────────────────────────────────────────────

    def _test_planner_agent(self) -> Dict:
        from laravel_modules.ai_brain.agent_pipeline import PlannerAgent
        p = PlannerAgent()
        result = p.run({"query": "total sales today"})
        assert "sales" in result["plan"]["domains"]
        return {"domains": result["plan"]["domains"]}

    def _test_validator_agent(self) -> Dict:
        from laravel_modules.ai_brain.agent_pipeline import ValidatorAgent
        v = ValidatorAgent()
        result = v.run({
            "insights": [],
            "analysis_results": [{"confidence": 0.8}],
            "recommendations": [],
            "plan": {},
        })
        assert result["is_valid"]
        return {"valid": True}

    def _test_narrator_agent(self) -> Dict:
        from laravel_modules.ai_brain.agent_pipeline import NarratorAgent
        n = NarratorAgent()
        result = n.run({
            "insights": [{"finding": "test", "confidence": 0.9}],
            "recommendations": [],
            "causes": [], "warnings": [],
            "plan": {"domains": ["general"], "analysis_types": ["detail"],
                     "time_period": "all_time", "complexity": "simple"},
            "is_valid": True, "confidence_adjusted": 0.8,
            "original_query": "test",
        })
        assert "narrative" in result
        return {"has_narrative": True}

    def _test_pipeline_plan_only(self) -> Dict:
        from laravel_modules.ai_brain.agent_pipeline import AgentPipeline
        p = AgentPipeline()
        result = p.process_plan_only("top products")
        assert "plan" in result
        return {"plan_created": True}

    # ── Memory Tests ─────────────────────────────────────────────────────

    def _test_short_term_memory(self) -> Dict:
        from laravel_modules.ai_brain.memory_service import ShortTermMemory
        stm = ShortTermMemory(max_entries=5)
        stm.remember("k1", "v1")
        assert stm.recall("k1") == "v1"
        return {"size": stm.size}

    def _test_memory_search(self) -> Dict:
        from laravel_modules.ai_brain.memory_service import ShortTermMemory
        stm = ShortTermMemory()
        stm.remember("sales_data", "100 items sold")
        results = stm.search("sales")
        assert len(results) >= 1
        return {"found": len(results)}

    def _test_context_stack(self) -> Dict:
        from laravel_modules.ai_brain.memory_service import ShortTermMemory
        stm = ShortTermMemory()
        stm.push_context({"query": "test"})
        ctx = stm.get_recent_context(1)
        assert len(ctx) == 1
        return {"stack_size": 1}

    # ── Embedding Tests ──────────────────────────────────────────────────

    def _test_hash_embedding(self) -> Dict:
        from laravel_modules.ai_brain.embedding_engine import HashEmbedder
        h = HashEmbedder(dim=128)
        vec = h.embed_tokens(["hello", "world"])
        assert len(vec) == 128
        return {"dim": 128}

    def _test_vector_index(self) -> Dict:
        from laravel_modules.ai_brain.embedding_engine import VectorIndex
        idx = VectorIndex(128)
        idx.insert("d1", [0.1] * 128, {"text": "test"})
        r = idx.search([0.1] * 128, top_k=1)
        assert r[0]["id"] == "d1"
        return {"size": idx.size}

    def _test_cosine_similarity(self) -> Dict:
        from laravel_modules.ai_brain.embedding_engine import cosine_similarity
        s = cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(s - 1.0) < 0.001
        return {"self_similarity": s}

    def _test_business_embedding(self) -> Dict:
        from laravel_modules.ai_brain.embedding_engine import BusinessEmbeddingEngine
        e = BusinessEmbeddingEngine(128)
        v = e.embed_text("cement 50kg bag")
        assert len(v) == 128
        return {"dim": 128}

    # ── RAG Tests ────────────────────────────────────────────────────────

    def _test_context_builder(self) -> Dict:
        from laravel_modules.ai_brain.rag_engine import ContextBuilder
        cb = ContextBuilder(500)
        ctx = cb.build([{"text": "fact1", "similarity": 0.9}], "query")
        assert ctx["total_chunks"] == 1
        return {"chunks": 1}

    def _test_fact_grounder(self) -> Dict:
        from laravel_modules.ai_brain.rag_engine import FactGrounder
        fg = FactGrounder()
        result = fg.check_grounding("answer", [{"text": "evidence"}])
        assert "grounding_score" in result
        return {"score": result["grounding_score"]}

    def _test_rag_augmented_query(self) -> Dict:
        from laravel_modules.ai_brain.rag_engine import RAGEngine
        rag = RAGEngine()
        result = rag.augmented_query("test query")
        assert "augmented_query" in result
        return {"augmented": True}

    # ── Integration Tests ────────────────────────────────────────────────

    def _test_transformer_to_pipeline(self) -> Dict:
        from laravel_modules.ai_brain.agent_pipeline import AgentPipeline
        from laravel_modules.ai_brain.transformer_core import TRANSFORMER_BRAIN
        p = AgentPipeline()
        assert p.reasoner._get_transformer() is not None
        return {"integrated": True}

    def _test_embedding_to_rag(self) -> Dict:
        from laravel_modules.ai_brain.rag_engine import RAGEngine
        from laravel_modules.ai_brain.embedding_engine import EMBEDDING_ENGINE
        rag = RAGEngine()
        assert rag._get_embedding_engine() is not None
        return {"integrated": True}

    def _test_memory_to_rag(self) -> Dict:
        from laravel_modules.ai_brain.rag_engine import RAGEngine
        from laravel_modules.ai_brain.memory_service import MEMORY_SERVICE
        rag = RAGEngine()
        assert rag._get_memory_service() is not None
        return {"integrated": True}

    # ── Smoke Tests ──────────────────────────────────────────────────────

    def _test_all_imports(self) -> Dict:
        """Quick check: all modules importable."""
        modules = {}
        try:
            from laravel_modules.ai_brain import transformer_core
            modules["transformer_core"] = True
        except:
            modules["transformer_core"] = False

        try:
            from laravel_modules.ai_brain import agent_pipeline
            modules["agent_pipeline"] = True
        except:
            modules["agent_pipeline"] = False

        try:
            from laravel_modules.ai_brain import memory_service
            modules["memory_service"] = True
        except:
            modules["memory_service"] = False

        try:
            from laravel_modules.ai_brain import embedding_engine
            modules["embedding_engine"] = True
        except:
            modules["embedding_engine"] = False

        try:
            from laravel_modules.ai_brain import rag_engine
            modules["rag_engine"] = True
        except:
            modules["rag_engine"] = False

        try:
            from laravel_modules.ai_brain import sales_deep_intelligence
            modules["sales_deep_intelligence"] = True
        except:
            modules["sales_deep_intelligence"] = False

        try:
            from laravel_modules.ai_brain import customer_debt_engine
            modules["customer_debt_engine"] = True
        except:
            modules["customer_debt_engine"] = False

        try:
            from laravel_modules.ai_brain import expense_intelligence
            modules["expense_intelligence"] = True
        except:
            modules["expense_intelligence"] = False

        try:
            from laravel_modules.ai_brain import business_health_engine
            modules["business_health_engine"] = True
        except:
            modules["business_health_engine"] = False

        assert all(modules.values()), f"Failed imports: {[k for k, v in modules.items() if not v]}"
        return modules

    def _test_singletons(self) -> Dict:
        """Verify all global singletons exist."""
        from laravel_modules.ai_brain.transformer_core import TRANSFORMER_BRAIN
        from laravel_modules.ai_brain.agent_pipeline import AGENT_PIPELINE
        from laravel_modules.ai_brain.memory_service import MEMORY_SERVICE
        from laravel_modules.ai_brain.embedding_engine import EMBEDDING_ENGINE
        from laravel_modules.ai_brain.rag_engine import RAG_ENGINE

        assert TRANSFORMER_BRAIN is not None
        assert AGENT_PIPELINE is not None
        assert MEMORY_SERVICE is not None
        assert EMBEDDING_ENGINE is not None
        assert RAG_ENGINE is not None
        return {"singletons": 5}

    # ── Run Methods ──────────────────────────────────────────────────────

    def run_all(self) -> Dict[str, Any]:
        """Run all test suites."""
        start = time.time()
        self.last_run_results = []

        for suite in self.suites:
            results = suite.run()
            self.last_run_results.extend(results)

        total = len(self.last_run_results)
        passed = sum(1 for r in self.last_run_results if r.passed)
        failed = total - passed
        elapsed = round(time.time() - start, 4)

        run_summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
            "execution_time": elapsed,
            "suites": len(self.suites),
            "results": [r.to_dict() for r in self.last_run_results],
            "failures": [r.to_dict() for r in self.last_run_results if not r.passed],
        }

        self.run_history.append(run_summary)
        return run_summary

    def run_smoke(self) -> Dict[str, Any]:
        """Run only smoke tests for a quick health check."""
        suite = next((s for s in self.suites if s.category == "smoke"), None)
        if not suite:
            return {"error": "No smoke suite found"}

        results = suite.run()
        return {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "results": [r.to_dict() for r in results],
        }

    def run_category(self, category: str) -> Dict[str, Any]:
        """Run tests by category (unit, integration, smoke, regression)."""
        suites = [s for s in self.suites if s.category == category]
        results = []
        for suite in suites:
            results.extend(suite.run())

        return {
            "category": category,
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "results": [r.to_dict() for r in results],
        }


# =============================================================================
AUTO_TEST_ENGINE = AutoTestEngine()
logger.info("Auto-Test Engine v1.0.0 — Ready with {0} suites.".format(len(AUTO_TEST_ENGINE.suites)))
