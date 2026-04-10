import numpy as np

class SignalCalibrator:
    def __init__(self, temperature=1.5):
        self.temperature = temperature     # Higher temperature leads to softer probabilities, while lower temperature makes the distribution sharper. A temperature of 1.0 means no change, while values greater than 1.0 will make the distribution softer and values less than 1.0 will make it sharper.

    def calibrate_confidence(self, confidence):
        """
        Temperature scaling (simulated for confidence values)
        """
        # Avoid log(0)
        confidence = np.clip(confidence, 1e-6, 1.0)   # We are using np.clip() to ensure that the confidence values are within a safe range for logarithmic transformation. By setting a lower bound of 1e-6, we prevent the confidence from being exactly zero, which would lead to an undefined logarithm. The upper bound of 1.0 ensures that confidence values do not exceed the maximum possible value for probabilities.

        # Convert to logit
        logit = np.log(confidence / (1 - confidence))  # The logit function transforms a probability (confidence) into a log-odds value. It is calculated as the natural logarithm of the ratio of the confidence to its complement (1 - confidence). This transformation allows us to apply temperature scaling in the log-odds space, which can help in adjusting the confidence values more effectively.

        # Apply temperature
        scaled_logit = logit / self.temperature  # By dividing the logit by the temperature, we can control the sharpness of the confidence distribution. A higher temperature will make the distribution softer (more uniform), while a lower temperature will make it sharper (more peaked around certain values).

        # Convert back to probability
        calibrated = 1 / (1 + np.exp(-scaled_logit))  

        return float(np.clip(calibrated, 0.0, 1.0))
    
    def normalize_entropy(self, entropy, mean, std):
        """
        Normalize entropy using dataset stats
        """
        return float((entropy - mean) / (std + 1e-6))
    
    def compute_trust(self, confidence, entropy_norm):
        """
        Improved trust formulation
        """
        entropy_scaled = np.tanh(entropy_norm)  # The np.tanh() function is used to create a smooth, non-linear relationship between the normalized entropy and the trust score. By applying the hyperbolic tangent function to the normalized entropy, we can ensure that the influence of entropy on trust is more gradual and less sensitive to extreme values. This helps in creating a more balanced trust score that takes into account both confidence and uncertainty in a more nuanced way.

        # normalize to [0,1]
        entropy_scaled = (entropy_scaled + 1) / 2  # The output of np.tanh() ranges from -1 to 1. By adding 1 and then dividing by 2, we can scale this output to a range of [0, 1]. This makes it easier to combine with the confidence score, which is also in the range of [0, 1], when calculating the final trust score.
        # Higher confidence + lower entropy = higher trust
        trust = confidence * (1 - entropy_scaled)   # The np.tanh() function is used to create a smooth, non-linear relationship between the normalized entropy and the trust score.   

        return float(np.clip(trust, 0.0, 1.0))