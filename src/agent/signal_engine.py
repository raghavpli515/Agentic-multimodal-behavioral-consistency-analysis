import numpy as np
# This module defines the SignalEngine class, which analyzes the outputs of video, audio, and text models to determine the dominant modality, compute confidence and entropy, and assess the reliability of the predictions.
class SignalEngine:

    def __init__(self):
        pass 

    def compute_entropy(self, probs):
        probs = np.array(probs)
        return -np.sum(probs * np.log(probs + 1e-10))

    def dominant_modality(self, video_probs, audio_probs, text_probs):
        max_video = max(video_probs)
        max_audio = max(audio_probs)
        max_text = max(text_probs)

        scores = {
            "video": max_video,
            "audio": max_audio,
            "text": max_text
        }

        return max(scores, key=scores.get) 

    def disagreement_score(self, video_probs, audio_probs, text_probs):
        v = np.array(video_probs)
        a = np.array(audio_probs)
        t = np.array(text_probs)

        d_va = np.mean(np.abs(v - a)) 
        d_vt = np.mean(np.abs(v - t))
        d_at = np.mean(np.abs(a - t))

        return float((d_va + d_vt + d_at) / 3)

    def reliability_level(self, confidence, entropy, disagreement):
        if confidence > 0.75 and entropy < 0.8 and disagreement < 0.2:
            return "HIGH"
        elif confidence > 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def analyze(self, video_probs, audio_probs, text_probs, fusion_probs):

        predicted_class = int(np.argmax(fusion_probs)) 
        confidence = float(max(fusion_probs)) 
        entropy = float(self.compute_entropy(fusion_probs))
        disagreement = self.disagreement_score(video_probs, audio_probs, text_probs)
        dominant = self.dominant_modality(video_probs, audio_probs, text_probs)
        reliability = self.reliability_level(confidence, entropy, disagreement)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "entropy": entropy,
            "disagreement_score": disagreement,
            "dominant_modality": dominant,
            "reliability": reliability
        }