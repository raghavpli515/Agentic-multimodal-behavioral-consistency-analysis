from .temporal_analyzer import TemporalAnalyzer
from .pattern_detector import PatternDetector
from .reasoning_engine import ReasoningEngine

class BehavioralAgent:
    def __init__(self,segments):
        self.segments = segments

    def analyze(self):
        temporal = TemporalAnalyzer(self.segments)
        trends = temporal.compute_trends()
        volatility = temporal.compute_volatility()

        pattern = PatternDetector(self.segments)
        patterns = {
            "instability": pattern.detect_instability(),
            "flip_rate": pattern.detect_prediction_flips(),
            "suspicious_segments": pattern.detect_suspicious_segments(),
            "drift": pattern.detect_drift()
        }

        engine = ReasoningEngine(trends, volatility, patterns)

        overall_trust = engine.compute_trust_score()
        reasoning = engine.generate_reasoning()

        return {
            "overall_trust": overall_trust,
            "trends": trends,
            "volatility": volatility,
            "patterns": patterns,
            "reasoning": reasoning
        }
        