# this script defines a memory manager for tracking the agent's behavioral states, trust levels, and entropy over time. It provides methods to update the memory with new segments, retrieve recent states, compute persistence of behavioral states, and summarize the memory for analysis.
from collections import deque
from librosa import segment
import numpy as np

class BehavioralMemoryManager:

    def __init__(self, max_memory=50):

        self.memory = deque(maxlen=max_memory)
        self.modal_conflict_history = []
        self.persistent_conflicts = 0

    def update(self, segment):

        self.memory.append(segment)
        if segment.get("modal_conflict", False):

            self.modal_conflict_history.append(
            segment["segment_id"]
            )


    def get_recent_states(self):

        return [m["behavior_label"] for m in self.memory]

    def get_trust_history(self):

        return [m["trust"] for m in self.memory]

    def get_entropy_history(self):

        return [m["entropy"] for m in self.memory]

    def get_modal_conflicts(self):

        conflicts = 0

        for m in self.memory:

            modal_preds = m.get("modal_predictions", {})

            preds = list(modal_preds.values())

            if len(set(preds)) > 1:
                conflicts += 1

        return conflicts

    def compute_behavioral_persistence(self):

        states = self.get_recent_states()

        persistence = {
            "stable_runs": 0,
            "uncertain_runs": 0,
            "inconsistent_runs": 0
        }

        current = None
        count = 0

        for s in states:

            if s == current:
                count += 1
            else:

                if current == "STABLE":
                    persistence["stable_runs"] = max(
                        persistence["stable_runs"],
                        count
                    )

                elif current == "UNCERTAIN":
                    persistence["uncertain_runs"] = max(
                        persistence["uncertain_runs"],
                        count
                    )

                elif current == "INCONSISTENT":
                    persistence["inconsistent_runs"] = max(
                        persistence["inconsistent_runs"],
                        count
                    )

                current = s
                count = 1

        return persistence

    def analyze_modal_conflicts(self):

        if len(self.modal_conflict_history) == 0:

            return {
                "persistent_conflict": False,
                "conflict_runs": 0
            }

        runs = 1

        for i in range(1, len(self.modal_conflict_history)):

            prev_seg = self.modal_conflict_history[i - 1]
            curr_seg = self.modal_conflict_history[i]

            if curr_seg - prev_seg <= 2:
                runs += 1

        persistent = runs >= 3

        return {
            "persistent_conflict": persistent,
            "conflict_runs": runs
        }


    def summarize_memory(self):

        trust = self.get_trust_history()

        entropy = self.get_entropy_history()
       
        modal_conflict_analysis = self.analyze_modal_conflicts()
        
        return {

            "avg_trust": float(np.mean(trust)) if trust else 0.0,

            "avg_entropy": float(np.mean(entropy)) if entropy else 0.0,

            "modal_conflicts": self.get_modal_conflicts(),

            "persistence": self.compute_behavioral_persistence(),

            "modal_conflict_analysis": modal_conflict_analysis
        }
    
