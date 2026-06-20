class ContextBuilder:

    def build(
        self,
        graph_result,
        trends,
        patterns,
        memory_summary,
        reasoning,
        persistent_modal_conflict
    ):

        return {

            "behavior_pattern":
                graph_result["behavior_pattern"],

            "dominant_state":
                graph_result["dominant_state"],

            "stability_score":
                graph_result["stability_score"],

            "confidence_trend":
                trends["confidence_trend"],

            "entropy_trend":
                trends["entropy_trend"],

            "trust_trend":
                trends["trust_trend"],

            "modal_conflicts":
                patterns["modal_conflicts"],

            "escalation_detected":
                patterns["escalation_detected"],

            "persistent_modal_conflict":
                persistent_modal_conflict,

            "memory_summary":
                memory_summary,

            "reasoning":
                reasoning
        }