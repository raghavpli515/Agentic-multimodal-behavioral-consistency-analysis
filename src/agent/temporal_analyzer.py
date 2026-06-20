
import numpy as np

class TemporalAnalyzer:
    def __init__(self, segments):
        self.segments = segments

    def extract_series(self):
        confidence = np.array([seg['confidence'] for seg in self.segments])
        entropy = np.array([seg['entropy'] for seg in self.segments])
        trust = np.array([seg['trust'] for seg in self.segments])
        predictions = np.array([seg['prediction'] for seg in self.segments])

        return confidence, entropy, trust, predictions
    
    def compute_trends(self):
        confidence, entropy, trust, _ = self.extract_series()

        return {
            "confidence_trend": np.polyfit(range(len(confidence)), confidence, 1)[0],   # np.polyfit() is a function from the NumPy library that fits a polynomial of a specified degree to a set of data points. In this case, we are fitting a linear polynomial (degree 1) to the confidence values over time. The function returns an array of coefficients for the fitted polynomial, and by accessing the first element [0], we get the slope of the line, which represents the trend of confidence over time. A positive slope indicates an increasing trend, while a negative slope indicates a decreasing trend in confidence across the segments.
            "entropy_trend": np.polyfit(range(len(entropy)), entropy, 1)[0],  # Similar to confidence_trend, this computes the slope of the linear fit for the entropy values over time, indicating whether the uncertainty in predictions is increasing or decreasing across the segments.
            "trust_trend": np.polyfit(range(len(trust)), trust, 1)[0]
        }
    
    def compute_volatility(self):
        confidence, entropy, _, _ = self.extract_series() # We are only interested in confidence and entropy for volatility analysis, as they directly relate to the stability of the predictions. The trust metric is derived from confidence and entropy, so it is not necessary to include it in the volatility calculation.

        return {
            "confidence_std": np.std(confidence),   # np.std() is a function from the NumPy library that calculates the standard deviation of a set of values. In this context, we are calculating the standard deviation of the confidence values across the segments to measure how much the confidence fluctuates over time. A higher standard deviation indicates greater volatility in confidence, while a lower standard deviation suggests more stable confidence levels across the segments.
            "entropy_std": np.std(entropy)  # Similar to confidence_std, this calculates the standard deviation of the entropy values across the segments, providing insight into how much the uncertainty in predictions varies over time. A higher standard deviation in entropy indicates more fluctuation in uncertainty, while a lower standard deviation suggests more consistent levels of uncertainty across the segments.
        }
    
    def compute_persistence(self):

        labels = [s['behavior_label'] for s in self.segments]

        persistent_uncertain = 0
        persistent_inconsistent = 0

        curr_uncertain = 0
        curr_inconsistent = 0

        for label in labels:

            if label == "UNCERTAIN":
                curr_uncertain += 1
                persistent_uncertain = max(
                    persistent_uncertain,
                    curr_uncertain
                )
            else:
                curr_uncertain = 0

            if label == "INCONSISTENT":
                curr_inconsistent += 1
                persistent_inconsistent = max(
                    persistent_inconsistent,
                    curr_inconsistent
                )
            else:
                curr_inconsistent = 0

        return {
            "persistent_uncertain": persistent_uncertain,
            "persistent_inconsistent": persistent_inconsistent
        }