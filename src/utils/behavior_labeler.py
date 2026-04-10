import numpy as np

class BehaviorLabeler:
    def __init__(self,segments):
        self.segments = segments

        self.conf_values = np.array([seg['confidence'] for seg in segments])
        self.entropy_values = np.array([seg['entropy'] for seg in segments])

        # Adaptive thresholds
        self.conf_mean = np.mean(self.conf_values)
        self.conf_std = np.std(self.conf_values)

        self.ent_mean = np.mean(self.entropy_values)
        self.ent_std = np.std(self.entropy_values)

    def normalize(self, confidence, entropy):
        # Safe normalization
        confidence = np.clip(confidence, 0.0, 1.0) 

        # Z score normalization
        entropy_norm = (entropy - self.ent_mean) / (self.ent_std + 1e-6)       # 1e-6 is added to the denominator to prevent division by zero in case the standard deviation is very small. 

        return confidence, entropy_norm 
    
    def assign_label(self, confidence, entropy, prev_prediction, curr_prediction):
        
        # Dynamic thresholds
    
        high_conf = self.conf_mean + 0.5 * self.conf_std
        low_conf = self.conf_mean - 0.5 * self.conf_std

        high_entropy = self.ent_mean + 0.5 * self.ent_std

        # STABLE
        if confidence > high_conf and entropy < self.ent_mean:
            return "STABLE"

        # INCONSISTENT
        if prev_prediction is not None and prev_prediction != curr_prediction:
            return "INCONSISTENT"
        
        # UNCERTAIN
        if entropy > high_entropy or confidence < low_conf:
            return "UNCERTAIN"
        
        return "STABLE"
        