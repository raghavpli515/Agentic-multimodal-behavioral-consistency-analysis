import numpy as np

class PatternDetector:
    def __init__(self, segments):
        self.segments = segments

    def detect_instability(self):
        confidence = np.array([s['confidence'] for s in self.segments])
        return float(np.std(np.diff(confidence)))  # np.diff() calculates the difference between consecutive confidence values, and np.std() computes the standard deviation of these differences to measure the instability in confidence across the segments. A higher value indicates greater fluctuations in confidence, while a lower value suggests more stable confidence levels throughout the segments.
    
    def detect_prediction_flips(self):
        preds = [s["prediction"] for s in self.segments]
        flips = sum(p1 != p2 for p1, p2 in zip(preds[:-1], preds[1:]))  # This line counts the number of times the predicted class changes between consecutive segments. By zipping the list of predictions with itself offset by one (preds[:-1] and preds[1:]), we can compare each prediction with the next one. The sum counts how many times these comparisons are not equal, which indicates a flip in the predicted class. A higher number of flips suggests more volatility in the predictions across the segments.
        return flips / len(preds)  # Normalizing the number of flips by the total number of predictions gives us a ratio that represents the frequency of prediction changes. This allows us to compare instability across different videos or segments of varying lengths, providing a more standardized measure of prediction volatility.
    

    def detect_suspicious_segments(self):
        suspicious = []
        for s in self.segments:
            if s['confidence'] < 0.4 and s['entropy'] > 1.3:
                suspicious.append(s['segment_id'])
        return suspicious  
    

    def detect_drift(self):
        confidence = np.array([s['confidence'] for s in self.segments])
        return float(confidence[-1] - confidence[0])  # This line calculates the drift in confidence by taking the difference between the last confidence value and the first confidence value in the segments. A positive value indicates an increase in confidence over time, while a negative value indicates a decrease. This measure helps us understand whether the model's confidence is improving or deteriorating across the segments, which can be an important indicator of concept drift or changes in the underlying data distribution.
            

    def detect_escalation(self):

        entropy = np.array([s['entropy'] for s in self.segments])
        trust = np.array([s['trust'] for s in self.segments])

        entropy_trend = np.polyfit(
            range(len(entropy)),
            entropy,
            1
        )[0]

        trust_trend = np.polyfit(
            range(len(trust)),
            trust,
            1
        )[0]

        if entropy_trend > 0.02 and trust_trend < -0.02:
            return True

        return False
    
    def detect_modal_conflicts(self):

        conflicts = []

        for s in self.segments:

            modal_preds = s.get(
                "modal_predictions",
                {}
            )

            if len(modal_preds) < 2:
                continue

            preds = list(modal_preds.values())

            unique_preds = set(preds)

            # ====================================
            # NO CONFLICT
            # ====================================

            if len(unique_preds) == 1:
                continue

            # ====================================
            # CONFLICT SEVERITY
            # ====================================

            severity = "LOW"

            if len(unique_preds) == 2:
                severity = "MEDIUM"

            if len(unique_preds) >= 3:
                severity = "HIGH"

            conflicts.append({

                "segment_id": s["segment_id"],

                "modal_predictions": modal_preds,

                "severity": severity,

                "trust": s.get("trust", 0),

                "confidence": s.get("confidence", 0)
            })

        return conflicts    