import sys
import os
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(ROOT_DIR)

from src.pipeline.interview_analyzer import InterviewAnalyzer
from agent.behavioral_agent import BehavioralAgent
from src.utils.behavior_labeler import BehaviorLabeler
from src.utils.signal_calibrator import SignalCalibrator

def run_system(video_path: str):

    print(f"[DEBUG] Received video path: {video_path}")

    analyzer = InterviewAnalyzer()
    timeline = analyzer.analyze(video_path)

    print(f"[DEBUG] Timeline output length: {len(timeline)}")

    
    #========================================
    #  COLLECT  RAW  SEGMENTS
    #========================================
    raw_segments = []

    for i, t in enumerate(timeline):
        raw_segments.append({
            "segment_id": i,
            "prediction": t.get("prediction"),
            "confidence": float(t.get("confidence")),
            "entropy": float(t.get("entropy")),
            "trust": float(t.get("trust")),
            "reliability": t.get("reliability")
        })

    #==========================================
    #  INITIALIZE  CALIBRATOR
    #==========================================
    calibrator = SignalCalibrator(temperature=1.5)

    # dataset stats
    entropies = [seg['entropy'] for seg in raw_segments]
    ent_mean = np.mean(entropies)
    ent_std = np.std(entropies)

    
    #========================================
    #  CALIBRATE  FIRST
    #========================================
    calibrated_segments = []

    for s in raw_segments:

        calibrated_conf = calibrator.calibrate_confidence(s['confidence'])
        entropy_norm = calibrator.normalize_entropy(s['entropy'], ent_mean, ent_std)
        calibrated_trust = calibrator.compute_trust(calibrated_conf, entropy_norm)

        calibrated_segments.append({
            **s,     # This syntax is used to create a new dictionary that includes all the key-value pairs from the original dictionary s, and then we can add or override specific keys with new values. 
            "raw_confidence": s["confidence"],
            "confidence": calibrated_conf,
            "entropy_norm": entropy_norm,
            "trust": calibrated_trust
        })

    #========================================
    #   LABELING
    #========================================    
    labeler = BehaviorLabeler(calibrated_segments)

    segments = []
    prev_prediction = None

    for s in calibrated_segments:

        behavior_label = labeler.assign_label(s["confidence"], s["entropy_norm"], prev_prediction, s["prediction"])
        s["behavior_label"] = behavior_label

        segments.append(s)
        prev_prediction = s['prediction']


    # ========================================
    #  AGENTIC REASONING LAYER
    # ========================================

    agent = BehavioralAgent(segments)
    agent_result = agent.analyze()


    # ========================================
    # FINAL TRUST DECISION (AGENT-DRIVEN)
    # ========================================

    agent_score = agent_result["overall_trust"]

    if agent_score > 0.7:
        overall = "HIGH"
    elif agent_score > 0.4:
        overall = "MEDIUM"
    else:
        overall = "LOW"

    
    # ========================================
    #  FINAL OUTPUT
    # ========================================
    final_output = {
        "segments": segments,
        "agent_analysis": agent_result,
        "overall_trust": overall,
    }

    return final_output

if __name__ == "__main__":    # This allows us to run this file directly for testing purposes,__name__ == "__main__": means this block will only execute if we run this file directly, and not when we import it as a module in another file.
    
    video_path = "dataset/baseline/baseline_q1_t1.mp4" 

    result = run_system(video_path)

    import json
    print(json.dumps(result, indent=2)) 