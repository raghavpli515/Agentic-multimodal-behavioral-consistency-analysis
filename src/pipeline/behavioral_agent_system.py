import json
from src.inference.video_processor import VideoProcessor
from src.inference.inference_engine import InferenceEngine
from src.agent.signal_engine import SignalEngine


class BehavioralAgentSystem:

    def __init__(self, llm_client=None):
        """
        llm_client: function that takes a prompt and returns LLM response.
        This keeps the architecture API-agnostic.
        """
        print("Initializing Behavioral Agent System...")

        self.processor = VideoProcessor()
        self.inference_engine = InferenceEngine()
        self.signal_engine = SignalEngine()

        self.llm_client = llm_client

        print("System Ready.")

    def analyze_video(self, video_path):

        print("Processing video...")

        # --------------------------------
        # Extract raw modalities
        # --------------------------------
        frames = self.processor.extract_frames(video_path)
        mfcc = self.processor.extract_audio_features(video_path)
        input_ids, attention_mask = self.processor.transcribe(video_path)

        # --------------------------------
        # Run multimodal inference
        # --------------------------------
        probs = self.inference_engine.predict(
            frames,
            mfcc,
            input_ids,
            attention_mask
        )

        # --------------------------------
        # Behavioral signal analysis
        # --------------------------------
        signals = self.signal_engine.analyze(
            probs["video_probs"],
            probs["audio_probs"],
            probs["text_probs"],
            probs["fusion_probs"]
        )

        # --------------------------------
        # LLM reasoning agent
        # --------------------------------
        report = None  # Initialize report variable to None to handle cases where llm_client is not provided. This ensures that the variable is defined and can be safely returned in the final output, even if the LLM reasoning step is skipped due to the absence of an llm_client.

        if self.llm_client is not None:

            prompt = f"""
You are an AI behavioral analysis assistant.

Given the following multimodal behavioral signals from an interview:

{json.dumps(signals, indent=2)}          # This JSON object contains the predicted class, confidence, entropy, disagreement score, dominant modality, and reliability level based on the analysis of video, audio, and text model outputs.
                                         
Generate a structured behavioral assessment including:

1. Summary of detected behavioral state
2. Interpretation of confidence and reliability
3. Explanation of modality contributions
4. Communication stability assessment
5. Suggestions for improvement

Respond in JSON format with fields:
summary
behavioral_analysis
reliability_comment
stability_score
recommendations
"""

            report = self.llm_client(prompt)

        # --------------------------------
        #  Return full system output
        # --------------------------------
        return {
            "model_outputs": probs,
            "signal_metrics": signals,
            "agent_report": report
        }