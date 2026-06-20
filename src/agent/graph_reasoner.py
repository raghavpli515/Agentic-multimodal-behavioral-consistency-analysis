import numpy as np
from collections import Counter, defaultdict

class GraphReasoner:
    def __init__(self, segments):
        self.segments = segments
        self.states = [s["behavior_label"] for s in segments]

    def compute_state_counts(self):
        counts = Counter(self.states)   # Count the occurrences of each state
        total = len(self.states)        # Total number of states

        return {
            state: count / total for state, count in counts.items()
        }

    def compute_transitions(self):

        transitions = defaultdict(int)

        for s1, s2 in zip(self.states[:-1], self.states[1:]):
            key = f"{s1} -> {s2}"
            transitions[key] += 1

        total = sum(transitions.values())

        return {
            k: float(v / total)
            for k, v in transitions.items()
        }
    
    def compute_stability_score(self):
        flips = sum(
            s1 != s2 for s1, s2 in zip(self.states[:-1], self.states[1:])
        )
        return 1- (flips / len(self.states))
    
    def dominant_state(self):
        return Counter(self.states).most_common(1)[0][0]  # .most_common(1) returns a list of the most common element and its count, [0][0] extracts the state from that list
    
    def infer_pattern(self, state_dist, transitions):
        
        stable = state_dist.get("STABLE", 0)
        uncertain = state_dist.get("UNCERTAIN", 0)
        inconsistent = state_dist.get("INCONSISTENT", 0)
        
        if stable > 0.6 and inconsistent < 0.1:
            return "Mostly stable behavior with minor uncertainity"
        
        if inconsistent > 0.3:
            return "Highly inconsistent behavior with frequent state changes"
        
        if uncertain > 0.5:
            return "Dominated by uncertainity with low confidence signals"
        
        if transitions.get("STABLE -> UNCERTAIN", 0) > 0.15:
            return "Generally stable but shows periodic uncertainity spikes"
        
        if transitions.get("UNCERTAIN -> STABLE", 0) > 0.15:
            return "Recovers well from uncertain states"
        
        if transitions.get("UNCERTAIN -> STABLE", 0) > 0.25:

            return (
                "Behavior demonstrates recovery capability "
                "from uncertain states"
            )
        
        return "Mixed behavioral pattern with moderate stability"
    

    def analyze(self):

        state_dist = self.compute_state_counts()
        transitions = self.compute_transitions()
        stability = self.compute_stability_score()
        dominant = self.dominant_state()

        pattern = self.infer_pattern(state_dist, transitions)

        return {
            "state_distribution": state_dist,
            "transitions": transitions,
            "dominant_state": dominant,
            "stability_score": stability,
            "behavior_pattern": pattern
        }