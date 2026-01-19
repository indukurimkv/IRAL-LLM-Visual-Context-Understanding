"""Client for interacting with OpenRouter API."""

import base64
import re
from typing import Optional, Tuple

from openai import OpenAI


PROMPT = """
You are inspecting an environment for anomalies and hazards. Use the following definitions:

    Hazard: any source of potential damage, harm, or adverse health effects on something or someone under certain conditions at work.

    Anomaly: A scene element that is out of place or unusual but not necessarily dangerous. It is something that breaks expected object–context relations (interposition, support, probability, position, or familiar size); appears in an unlikely place or state; rests on an impossible surface; has an unrealistic size relative to nearby objects; or appears different from its default expected appearance.

You MUST obey this strict output format:

    <code>: <one-sentence explanation>

Where <code> is ONLY one of:
    00 = Safe
    10 = Anomalous
    01 = Dangerous
    11 = Anomalous Dangerous

RULES:
    - The FIRST characters in your output MUST be the 2-digit code.
    - Do NOT add any text, words, markdown, or labels before the code.
    - After the code, type a colon and a concise explanation.
    - no bullet points.
"""


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        # Store API credentials and model identifier
        self.api_key = api_key
        self.model = model
        # Initialize OpenAI client with OpenRouter endpoint (OpenAI-compatible API)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
    
    def classify_image(self, image_data: bytes, image_format: str = "png") -> Tuple[Optional[str], str]:
        """
        Query OpenRouter API for image classification.
        
        Returns: (predicted_code, full_response)
        """
        try:
            # Encode image bytes to base64 string for data URL format
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            # Create data URL for embedding image in API request
            image_data_url = f"data:image/{image_format};base64,{image_base64}"
            
            # Make vision API call with text prompt and image
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url}
                            }
                        ]
                    }
                ],
                max_tokens=200  # Limit response length since we only need 2 digits
            )
            
            # Extract model's text response
            full_response = response.choices[0].message.content.strip()
            
            # Parse 2-digit code from response text
            predicted_code = self._extract_code(full_response)
            
            return predicted_code, full_response
            
        except Exception as e:
            print(f"Error querying OpenRouter API: {e}")
            return None, str(e)
    
    @staticmethod
    def _extract_code(response: str) -> Optional[str]:
        """Extract 2-digit code from model response."""
        # Use regex to find first 2-digit number (word boundaries ensure exact match)
        match = re.search(r'\b\d{2}\b', response)
        if match:
            # Return the matched 2-digit code
            return match.group(0)
        return None
