

class ReasoningEngine:
    def __init__(self,trends,volatility,patterns):
        self.trends = trends
        self.volatility = volatility
        self.patterns = patterns


    def compute_trust_score(self):
        score  = 1.0

        # Penalize instability
        score -= self.volatility['confidence_std'] * 0.5

        # Penalize entropy growth
        if self.trends['entropy_trend'] > 0:
            score -= 0.2

        # Penalize prediction flips
        score -= self.patterns['flip_rate'] * 0.5

        # Penalize suspicious segments
        score -= len(self.patterns['suspicious_segments']) * 0.01

        return max(score, 0.0)
    
    def generate_reasoning(self):
        reasons = []

        if self.patterns['flip_rate'] > 0.3:
            reasons.append("Frequent prediction changes detected")

        if self.volatility['confidence_std'] > 0.2:
            reasons.append("High confidence instability observed")

        if self.trends['entropy_trend'] > 0:
            reasons.append(f"Entropy increasing (trend = {self.trends['entropy_trend']:.4f}) indicating rising uncertainity")

        if len(self.patterns['suspicious_segments']) > 5:
            reasons.append(f"{len(self.patterns['suspicious_segments'])} segments flagged as suspicious due to low confidence and high entropy")

        return reasons
            