# This script defines a NarrativeReasoner class that utilizes the OpenAI API to analyze structured behavioral contexts and generate comprehensive reports. The generate_report method takes in a behavioral context, constructs a prompt with specific instructions for analysis, and returns a detailed report covering various aspects of the behavior, such as trajectory, stability, trust evolution, modal conflicts, escalation, and consistency.
from openai import OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()

class NarrativeReasoner:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def generate_report(self, context):

        prompt = f"""
    You are an advanced trust-aware behavioral reasoning AI.

    Your task is NOT emotion recognition.

    Your task is:
    behavioral consistency analysis
    under uncertainty and multimodal disagreement.

    You must reason ONLY from the provided data.

    Do NOT:
    - diagnose mental conditions
    - infer intentions
    - speculate beyond evidence

    ====================================================
    AVAILABLE CONTEXT
    ====================================================

    {json.dumps(context, indent=2)}

    ====================================================
    REASONING OBJECTIVES
    ====================================================

    Analyze:

    1. Temporal Behavioral Evolution
    - Is behavior stabilizing or deteriorating?
    - Is trust increasing or decreasing?
    - Is uncertainty persistent?

    2. Cross-Modal Congruence
    - Do modalities agree?
    - Are there modality contradictions?
    - Are contradictions isolated or persistent?

    3. Behavioral Stability
    - Are predictions fluctuating heavily?
    - Is instability temporary or sustained?
    - Does the subject recover from uncertain states?

    4. Trust Interpretation
    - Is low trust caused by:
        - uncertainty?
        - modality disagreement?
        - instability?
        - volatility?

    5. Escalation Analysis
    - Is behavioral inconsistency escalating?
    - Are uncertainty spikes increasing?

    6. Final Behavioral Consistency Assessment
    Provide:
    - overall consistency interpretation
    - trust interpretation
    - behavioral reliability interpretation

    7. Longitudinal Cross-Modal Incongruence
    - Are modality contradictions persistent?
    - Do contradictions repeat over time?
    - Does behavioral inconsistency appear sustained?

    Interpret persistent modality disagreement as a
    behavioral reliability signal rather than isolated noise.

    ====================================================
    OUTPUT FORMAT
    ====================================================

    Generate:

    1. Behavioral trajectory summary

    2. Cross-modal consistency analysis

    3. Stability and volatility interpretation

    4. Trust evolution interpretation

    5. Escalation and uncertainty analysis

    6. Final behavioral consistency assessment

    Be analytical, cautious, and evidence-grounded.
    """

        response = self.client.chat.completions.create(

            model="gpt-4.1-mini",

            messages=[
                {
                    "role": "system",
                    "content":
                    "You are a multimodal behavioral reasoning system focused on trust-aware analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3
        )

        return response.choices[0].message.content