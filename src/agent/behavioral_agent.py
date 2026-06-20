from .temporal_analyzer import TemporalAnalyzer
from .pattern_detector import PatternDetector
from .reasoning_engine import ReasoningEngine
from .graph_reasoner import GraphReasoner
from .memory_manager import BehavioralMemoryManager
from .context_builder import ContextBuilder
from .narrative_reasoner import NarrativeReasoner


class BehavioralAgent:

    def __init__(self, segments):

        self.segments = segments

    def analyze(self):

        # =========================================
        # TEMPORAL ANALYSIS
        # =========================================

        temporal = TemporalAnalyzer(self.segments)

        trends = temporal.compute_trends()

        volatility = temporal.compute_volatility()

        persistence = temporal.compute_persistence()

        # Convert numpy -> native python

        trends = {
            k: float(v)
            for k, v in trends.items()
        }

        volatility = {
            k: float(v)
            for k, v in volatility.items()
        }

        persistence = {
            k: int(v) if isinstance(v, int)
            else float(v)
            for k, v in persistence.items()
        }

        # =========================================
        # GRAPH REASONING
        # =========================================

        graph = GraphReasoner(self.segments)

        graph_result = graph.analyze()

        # Fix nested values

        graph_result["stability_score"] = float(
            graph_result["stability_score"]
        )

        graph_result["state_distribution"] = {

            k: float(v)

            for k, v in graph_result[
                "state_distribution"
            ].items()
        }

        graph_result["transitions"] = {

            k: float(v)

            for k, v in graph_result[
                "transitions"
            ].items()
        }

        # =========================================
        # PATTERN DETECTION
        # =========================================

        pattern = PatternDetector(self.segments)

        patterns = {

            "instability": float(
                pattern.detect_instability()
            ),

            "flip_rate": float(
                pattern.detect_prediction_flips()
            ),

            "suspicious_segments":
                pattern.detect_suspicious_segments(),

            "drift": float(
                pattern.detect_drift()
            ),

            "modal_conflicts":
                pattern.detect_modal_conflicts(),

            "escalation_detected":
                bool(pattern.detect_escalation())
        }

        # =========================================
        # BEHAVIORAL RECOVERY DETECTION
        # =========================================

        recovery_detected = (

            graph_result["transitions"].get(
                "UNCERTAIN -> STABLE",
                0
            ) > 0.25
        )

        patterns["behavioral_recovery"] = (
            recovery_detected
        )

        # =========================================
        # REASONING ENGINE
        # =========================================

        engine = ReasoningEngine(
            trends,
            volatility,
            patterns
        )

        overall_trust = float(
            engine.compute_trust_score()
        )

        reasoning = engine.generate_reasoning()

        # =========================================
        # MEMORY MANAGER
        # =========================================

        memory = BehavioralMemoryManager()

        for s in self.segments:

            memory.update(s)

        memory_summary = (
            memory.summarize_memory()
        )

        persistent_modal_conflict = (

            memory_summary[
                "modal_conflict_analysis"
            ]
        )

        # =========================================
        # CONTEXT BUILDING
        # =========================================

        builder = ContextBuilder()

        context = builder.build(

            graph_result,

            trends,

            patterns,

            memory_summary,

            reasoning,

            persistent_modal_conflict
        )

        # =========================================
        # NARRATIVE REASONING
        # =========================================

        narrative_engine = NarrativeReasoner()

        narrative = (
            narrative_engine.generate_report(
                context
            )
        )

        # =========================================
        # FINAL OUTPUT
        # =========================================

        return {

            "overall_trust": overall_trust,

            "trends": trends,

            "volatility": volatility,

            "patterns": patterns,

            "reasoning": reasoning,

            "graph_analysis": graph_result,

            "persistence": persistence,

            "memory_summary": memory_summary,

            "context": context,

            "narrative_report": narrative
        }