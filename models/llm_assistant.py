"""
LLM Assistant for Deepfake Detection - Powered by DeepSeek API
----------------------------------------------------------------
Explains deepfake detection results in plain English.
"""

import json
import os
from typing import Optional

from openai import OpenAI


SYSTEM_PROMPT = """You are an AI assistant specialising in deepfake detection and digital media forensics.

Your role:
1. Explain deepfake detection results in simple, non-technical language.
2. Describe what visual artifacts or patterns were detected.
3. Help users understand how deepfakes are created and detected.
4. Provide guidance on how to spot deepfakes manually.
5. Explain the confidence scores and attention maps from the AI models.

When explaining results:
- If the image is classified as FAKE, explain what kinds of artifacts deepfakes typically show
  (blurring around edges, inconsistent lighting, unnatural skin texture, mismatched eyes)
- If the image is classified as REAL, explain why it passed the authenticity checks
- Reference the attention map to explain which facial regions the AI focused on
- Mention the confidence score and what it means

Key concepts you should be able to explain:
- CNN (Convolutional Neural Network): Analyzes the image pixel-by-pixel for artifacts
- LSTM (Long Short-Term Memory): Scans across face regions to find spatial inconsistencies
- Attention Map: Highlights which parts of the face the AI found most suspicious/important
- Transfer Learning: Using knowledge from millions of images to detect deepfakes

IMPORTANT: Always remind users that no AI detector is 100% accurate.
Deepfake technology is constantly evolving, and false positives/negatives can occur.
"""


class DeepfakeAssistant:
    """DeepSeek-powered assistant for explaining deepfake detection results."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ):
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self._base_url = base_url
        self.client = OpenAI(api_key=self._api_key, base_url=base_url) if self._api_key else None
        self.conversation_history = []
        self.model = model

    def set_api_key(self, api_key: str):
        self._api_key = api_key
        self.client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _require_client(self):
        if self.client is None:
            raise ValueError(
                "DeepSeek API key not set. Configure DEEPSEEK_API_KEY "
                "or paste the key in the app sidebar."
            )
        return self.client

    def set_detection_context(self, cnn_result: dict, lstm_result: dict):
        """Inject detection results as context for the conversation."""
        context = f"""
[DEEPFAKE DETECTION ANALYSIS RESULTS]

CNN Individual Frame Analysis:
  - Prediction: {cnn_result.get('prediction', 'Unknown')}
  - Confidence: {cnn_result.get('confidence', 0) * 100:.1f}%
  - Real Probability: {cnn_result.get('prob_real', 0) * 100:.1f}%
  - Fake Probability: {cnn_result.get('prob_fake', 0) * 100:.1f}%

CNN-LSTM Spatial Analysis:
  - Prediction: {lstm_result.get('prediction', 'Unknown')}
  - Confidence: {lstm_result.get('confidence', 0) * 100:.1f}%
  - Most Suspicious Region: {lstm_result.get('suspicious_region', 'N/A')}
  - Attention Focus: {lstm_result.get('attention_description', 'N/A')}

Please explain these results to the user.
"""
        self.conversation_history = [
            {"role": "user", "content": context},
            {
                "role": "assistant",
                "content": (
                    "I've reviewed the deepfake analysis results from both AI models. "
                    "I can see the CNN and CNN-LSTM predictions along with the attention data. "
                    "I'm ready to explain the findings. What would you like to know?"
                ),
            },
        ]

    def chat(self, user_message: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_message})
        client = self._require_client()

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self.conversation_history,
            ],
        )

        reply = response.choices[0].message.content or ""
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def generate_report(self, cnn_result: dict, lstm_result: dict) -> str:
        """Generate a one-shot analysis report."""
        prompt = f"""
Generate a brief deepfake analysis report based on these results:

CNN Result: {json.dumps(cnn_result)}
CNN-LSTM Result: {json.dumps(lstm_result)}

Structure:
1. Verdict (1 sentence)
2. What was detected
3. Confidence assessment
4. Limitations disclaimer
"""
        client = self._require_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def reset_conversation(self):
        self.conversation_history = []


if __name__ == "__main__":
    cnn_result = {"prediction": "Fake", "confidence": 0.89, "prob_fake": 0.89, "prob_real": 0.11}
    lstm_result = {"prediction": "Fake", "confidence": 0.92, "suspicious_region": "right jawline"}

    assistant = DeepfakeAssistant()
    assistant.set_detection_context(cnn_result, lstm_result)
    print("Deepfake Assistant ready. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        print(f"\nAssistant: {assistant.chat(user_input)}\n")
