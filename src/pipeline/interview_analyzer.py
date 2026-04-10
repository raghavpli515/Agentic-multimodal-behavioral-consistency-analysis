# Interview analyzer.py

import torch
import numpy as np
import os
import gc
from src.pipeline.temporal_segmenter import segment_video
from src.inference.inference_engine import InferenceEngine
from src.preprocess.audio_utils import extract_audio_segment, compute_mfcc
from src.preprocess.text_utils import transcribe_segment, encode_text

class InterviewAnalyzer:
    def __init__(self):
        self.engine = InferenceEngine()

    def analyze(self, video_path):
        print("Starting temporal analysis...")

        segments = segment_video(video_path, segment_seconds=3, fps=8)

        timeline_predictions = []
        print(f"[DEBUG] Total segments: {len(segments)}")

        for i, segment in enumerate(segments):
            print(f"Processing segment {i}...")

            # Extract data from segment
            frames = segment["frames"]
            start = segment["start"]
            end = segment["end"]

            # VIDEO
            video_tensor = self.engine.preprocess_frames(frames)

            # AUDIO 
            audio_tensor = None
            audio_path = None

            try:
                audio_path = extract_audio_segment(video_path, start, end)
                mfcc = compute_mfcc(audio_path)
                mfcc = mfcc.T  # Transpose to (time_frames, n_mfcc) for model input
                audio_tensor = torch.tensor(mfcc).unsqueeze(0).float()  #.unsqueeze(0) adds a batch dimension to the tensor, making it compatible with the expected input shape of the model. The .float() method converts the tensor to a floating-point data type, which is typically required for input to neural networks.

            except Exception as e:
                print(f"[WARNING] Audio failed for segment {i}: {e}")
                audio_tensor = None

            # TEXT
            text_ids = None  
            text_mask = None  

            try:
                text = transcribe_segment(audio_path)
                
                if text.strip() != "":     #.strip() removes any leading or trailing whitespace from the text. This is important because sometimes the transcription might return an empty string or just whitespace if there was no discernible speech in the audio segment. By checking if text.strip() is not empty, we can avoid unnecessary processing of empty text and ensure that we only attempt to encode meaningful transcriptions for our multimodal analysis pipeline.
                    
                    text_ids, text_mask = encode_text(text)

            except Exception as e:
                print(f"[WARNING] Text failed for segment {i}: {e}")
            
            #  MULTIMODAL PREDICTION
            output = self.engine.predict(
                video_tensor,
                audio_mfcc=audio_tensor,
                text_ids=text_ids,
                text_mask=text_mask

            )

            probs = output["fusion_probs"]

            # Prediction
            predicted_class = int(np.argmax(probs))
            confidence = float(np.max(probs))

            # Entropy
            probs_np = np.array(probs)
            entropy = -np.sum(probs_np * np.log(probs_np + 1e-9))

            max_entropy = np.log(len(probs_np))
            normalized_entropy = entropy / max_entropy

            # Trust score
            trust = confidence * (1 - normalized_entropy)

            # Reliability
            if trust > 0.5:
                reliability = "HIGH"
            elif trust > 0.3:
                reliability = "MEDIUM"
            else:
                reliability = "LOW"

            # Store
            timeline_predictions.append({
                "segment": i,
                "prediction": predicted_class,
                "confidence": confidence,
                "entropy": float(entropy),
                "trust": float(trust),
                "reliability": reliability
            })

            #  Cleanup temp audio file
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

            # cleanup
            del video_tensor
            del output
            if audio_tensor is not None:
                del audio_tensor

            torch.cuda.empty_cache()
            gc.collect()

        print(f"[DEBUG] Analysis complete. Total predictions: {len(timeline_predictions)}")
        return timeline_predictions