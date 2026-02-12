"""
SephlightyAI Self-Learning Engine
Author: Antigravity AI
Version: 1.0.0

Continuous learning system that improves over time through:
- User feedback loops (accept/reject decisions)
- Confidence calibration (adjust scores based on accuracy)
- Pattern learning (detect recurring user questions)
- Performance tracking (response quality over time)
"""

import logging
import time
import math
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

logger = logging.getLogger("SELF_LEARNING")
logger.setLevel(logging.INFO)


# =============================================================================
# 1. FEEDBACK TRACKER
# =============================================================================

class FeedbackTracker:
    """
    Tracks user feedback on AI responses: accepts, rejects, corrections.
    Used for confidence calibration and improving future answers.
    """

    def __init__(self):
        self.feedback_log: List[Dict[str, Any]] = []
        self.domain_accuracy: Dict[str, List[bool]] = defaultdict(list)

    def record_feedback(self, domain: str, query: str,
                        answer: str, confidence: float,
                        accepted: bool,
                        correction: Optional[str] = None) -> None:
        """Record user feedback on an AI response."""
        self.feedback_log.append({
            "domain": domain,
            "query": query,
            "answer_preview": answer[:200],
            "confidence": confidence,
            "accepted": accepted,
            "correction": correction,
            "timestamp": time.time(),
        })
        self.domain_accuracy[domain].append(accepted)

        logger.info(f"Feedback recorded: domain={domain}, accepted={accepted}")

    def get_domain_accuracy(self, domain: str) -> float:
        """Get historical accuracy for a domain."""
        records = self.domain_accuracy.get(domain, [])
        if not records:
            return 0.8  # default
        return sum(records) / len(records)

    def get_overall_accuracy(self) -> float:
        """Get overall accuracy across all domains."""
        all_records = [v for vals in self.domain_accuracy.values() for v in vals]
        if not all_records:
            return 0.8
        return sum(all_records) / len(all_records)

    def get_weak_domains(self, threshold: float = 0.6) -> List[str]:
        """Find domains where accuracy is below threshold."""
        weak = []
        for domain, records in self.domain_accuracy.items():
            if records and (sum(records) / len(records)) < threshold:
                weak.append(domain)
        return weak


# =============================================================================
# 2. CONFIDENCE CALIBRATOR
# =============================================================================

class ConfidenceCalibrator:
    """
    Calibrates AI confidence scores based on historical accuracy.
    If the AI is consistently overconfident in a domain, scores are adjusted down.
    """

    def __init__(self, feedback_tracker: FeedbackTracker):
        self.feedback = feedback_tracker
        self.calibration_factors: Dict[str, float] = {}

    def calibrate(self, raw_confidence: float,
                   domain: str = "general") -> float:
        """Adjust raw confidence based on historical accuracy."""
        domain_accuracy = self.feedback.get_domain_accuracy(domain)

        # If AI was overconfident, reduce future confidence
        if domain_accuracy < 0.5:
            factor = 0.7
        elif domain_accuracy < 0.7:
            factor = 0.85
        elif domain_accuracy < 0.9:
            factor = 0.95
        else:
            factor = 1.0

        self.calibration_factors[domain] = factor
        calibrated = raw_confidence * factor

        return round(min(1.0, max(0.0, calibrated)), 4)

    def get_calibration_report(self) -> Dict[str, Any]:
        """Report on calibration factors per domain."""
        return {
            "calibration_factors": self.calibration_factors,
            "overall_accuracy": self.feedback.get_overall_accuracy(),
            "weak_domains": self.feedback.get_weak_domains(),
        }


# =============================================================================
# 3. PATTERN LEARNER
# =============================================================================

class PatternLearner:
    """
    Detects recurring question patterns and caches common answers.
    Reduces response time for frequently asked questions.
    """

    def __init__(self, max_patterns: int = 500):
        self.max_patterns = max_patterns
        self.query_patterns: Counter = Counter()
        self.cached_answers: Dict[str, Dict[str, Any]] = {}
        self.domain_patterns: Dict[str, Counter] = defaultdict(Counter)

    def observe_query(self, query: str, domain: str = "general",
                      answer: str = "", confidence: float = 0.0) -> None:
        """Observe a query and learn from it."""
        normalized = self._normalize_query(query)
        self.query_patterns[normalized] += 1
        self.domain_patterns[domain][normalized] += 1

        # Cache if high confidence and frequently asked
        if confidence >= 0.8 and self.query_patterns[normalized] >= 3:
            self.cached_answers[normalized] = {
                "answer": answer,
                "confidence": confidence,
                "domain": domain,
                "hit_count": self.query_patterns[normalized],
                "cached_at": time.time(),
            }

        # Evict if too many patterns
        if len(self.query_patterns) > self.max_patterns:
            least_common = self.query_patterns.most_common()[-(self.max_patterns // 5):]
            for pattern, _ in least_common:
                del self.query_patterns[pattern]
                self.cached_answers.pop(pattern, None)

    def get_cached_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if we have a cached answer for this query."""
        normalized = self._normalize_query(query)
        cached = self.cached_answers.get(normalized)

        if cached:
            # Check staleness (expire after 24 hours)
            age = time.time() - cached.get("cached_at", 0)
            if age > 86400:
                del self.cached_answers[normalized]
                return None
            cached["hit_count"] += 1
            return cached

        return None

    def get_top_questions(self, domain: Optional[str] = None,
                          limit: int = 10) -> List[Dict]:
        """Get most frequently asked questions."""
        if domain:
            patterns = self.domain_patterns.get(domain, Counter())
        else:
            patterns = self.query_patterns

        return [
            {"query": q, "count": c}
            for q, c in patterns.most_common(limit)
        ]

    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching."""
        import re
        q = query.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)
        q = re.sub(r'\s+', ' ', q)
        return q


# =============================================================================
# 4. PERFORMANCE TRACKER
# =============================================================================

class PerformanceTracker:
    """
    Tracks AI response quality metrics over time.
    Monitors: response times, confidence trends, accuracy by domain.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.response_times: List[float] = []
        self.confidence_history: List[float] = []
        self.domain_metrics: Dict[str, Dict[str, List]] = defaultdict(
            lambda: {"times": [], "confidences": [], "accepted": []}
        )

    def track_response(self, domain: str, response_time: float,
                       confidence: float,
                       accepted: Optional[bool] = None) -> None:
        """Track a single AI response."""
        self.response_times.append(response_time)
        self.confidence_history.append(confidence)

        self.domain_metrics[domain]["times"].append(response_time)
        self.domain_metrics[domain]["confidences"].append(confidence)
        if accepted is not None:
            self.domain_metrics[domain]["accepted"].append(accepted)

        # Trim to max history
        if len(self.response_times) > self.max_history:
            self.response_times = self.response_times[-self.max_history:]
            self.confidence_history = self.confidence_history[-self.max_history:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance metrics."""
        if not self.response_times:
            return {"status": "no_data"}

        return {
            "total_responses": len(self.response_times),
            "avg_response_time": round(sum(self.response_times) / len(self.response_times), 4),
            "avg_confidence": round(sum(self.confidence_history) / len(self.confidence_history), 4),
            "p95_response_time": round(sorted(self.response_times)[int(len(self.response_times) * 0.95)], 4),
            "domains_tracked": list(self.domain_metrics.keys()),
        }

    def get_domain_performance(self, domain: str) -> Dict[str, Any]:
        """Get performance metrics for a specific domain."""
        metrics = self.domain_metrics.get(domain)
        if not metrics or not metrics["times"]:
            return {"status": "no_data"}

        accepted = metrics["accepted"]
        accuracy = sum(accepted) / len(accepted) if accepted else None

        return {
            "responses": len(metrics["times"]),
            "avg_response_time": round(sum(metrics["times"]) / len(metrics["times"]), 4),
            "avg_confidence": round(sum(metrics["confidences"]) / len(metrics["confidences"]), 4),
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
        }


# =============================================================================
# 5. UNIFIED SELF-LEARNING ENGINE
# =============================================================================

class SelfLearningEngine:
    """
    Unified self-learning system combining feedback, calibration,
    pattern learning, and performance tracking.
    """

    def __init__(self):
        self.feedback = FeedbackTracker()
        self.calibrator = ConfidenceCalibrator(self.feedback)
        self.pattern_learner = PatternLearner()
        self.performance = PerformanceTracker()
        logger.info("SelfLearningEngine initialized: 4 subsystems active.")

    def on_response(self, domain: str, query: str,
                    answer: str, confidence: float,
                    response_time: float) -> None:
        """Called after every AI response to update learning state."""
        # Learn patterns
        self.pattern_learner.observe_query(query, domain, answer, confidence)

        # Track performance
        self.performance.track_response(domain, response_time, confidence)

    def on_feedback(self, domain: str, query: str,
                    answer: str, confidence: float,
                    accepted: bool,
                    correction: Optional[str] = None) -> None:
        """Called when user provides feedback on an AI response."""
        self.feedback.record_feedback(
            domain, query, answer, confidence, accepted, correction
        )
        self.performance.track_response(domain, 0, confidence, accepted)

    def calibrate_confidence(self, raw_confidence: float,
                              domain: str = "general") -> float:
        """Get calibrated confidence based on historical accuracy."""
        return self.calibrator.calibrate(raw_confidence, domain)

    def check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Check if there's a cached answer for this query."""
        return self.pattern_learner.get_cached_answer(query)

    def get_learning_report(self) -> Dict[str, Any]:
        """Get comprehensive learning report."""
        return {
            "performance": self.performance.get_performance_summary(),
            "calibration": self.calibrator.get_calibration_report(),
            "top_questions": self.pattern_learner.get_top_questions(limit=10),
            "cache_size": len(self.pattern_learner.cached_answers),
            "total_feedback": len(self.feedback.feedback_log),
            "weak_domains": self.feedback.get_weak_domains(),
        }


# =============================================================================
SELF_LEARNING_ENGINE = SelfLearningEngine()
logger.info("Self-Learning Engine v1.0.0 — Continuous learning active.")
