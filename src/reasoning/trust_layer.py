import numpy as np
import torch
import torch.nn.functional as F


class TrustScorer:

    def __init__(self):

        self.high_conf_threshold = 0.75
        self.low_entropy_threshold = 0.8

    # =====================================
    # SOFTMAX
    # =====================================
    def get_probs(self, logits):

        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()[0]

    # =====================================
    # CONFIDENCE
    # =====================================
    def confidence(self, probs):

        return float(np.max(probs))

    # =====================================
    # ENTROPY
    # =====================================
    def entropy(self, probs):

        probs = np.clip(probs, 1e-9, 1.0)

        return float(-np.sum(probs * np.log(probs)))

    # =====================================
    # MODAL AGREEMENT
    # =====================================
    def modal_agreement(self, audio_probs, text_probs, video_probs):

        audio_pred = np.argmax(audio_probs)
        text_pred = np.argmax(text_probs)
        video_pred = np.argmax(video_probs)

        preds = [audio_pred, text_pred, video_pred]

        unique = len(set(preds))

        # =====================================
        if unique == 1:
            return 1.0

        # two same
        elif unique == 2:
            return 0.66

        # all different
        return 0.33

    # =====================================
    # FINAL TRUST SCORE
    # =====================================
    def compute(
        self,
        fusion_probs,
        audio_probs,
        text_probs,
        video_probs
    ):

        confidence = self.confidence(fusion_probs)

        entropy = self.entropy(fusion_probs)

        agreement = self.modal_agreement(
            audio_probs,
            text_probs,
            video_probs
        )

        # normalize entropy
        entropy_score = np.exp(-entropy)

        # weighted trust score
        trust = (
            0.5 * confidence +
            0.3 * agreement +
            0.2 * entropy_score
        )

        # behavioral label
        if trust >= 0.75:
            reliability = "STABLE"

        elif trust >= 0.5:
            reliability = "UNCERTAIN"

        else:
            reliability = "INCONSISTENT"

        return {
            "confidence": confidence,
            "entropy": entropy,
            "agreement": agreement,
            "trust": float(trust),
            "reliability": reliability
        } 
