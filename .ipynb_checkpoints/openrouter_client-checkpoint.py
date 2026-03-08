"""Client for interacting with OpenRouter API."""

import base64
import re
from typing import Optional, Tuple

from openai import OpenAI


PROMPT = """
PROMPT: You are inspecting an environment for anomalies and hazards to help first responders. Use the following definitions. Assume the image is taken from the point of view of a person physically present in the scene. Hazard: A scene is hazardous if normal interaction would likely cause immediate physical harm, and the danger is unavoidable or easy to miss. Anomaly: A scene is anomalous if any visible element clearly violates normal expectations for this environment (location, arrangement, state, or use). Safe: Safe scenes include normal, expected human activities in the specifically given appropriate environment, provided no immediate physical hazard is present. Before predicting the code, internally reason through these steps. (1) Identify any real physically present element that could cause immediate physical harm without hypothesizing. (2) If harm is likely and real, determine whether it is unavoidable or easy to miss. (3) Independently determine whether any element violates normal expectations for this environment. (4) Decide whether anomaly and hazard are present independently. You MUST obey this strict output format: <a brief, explicit justification for the final decision>: <code> Where <code> is ONLY one of: 00 Safe 10 Anomalous 01 = Hazardous 11 = Anomalous Hazardous - No extra paragraphs, no bullet points, no additional commentary.
"""


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet", prompt: str = PROMP):
        # Store API credentials and model identifier
        self.api_key = api_key
        self.model = model
        # Initialize OpenAI client with OpenRouter endpoint (OpenAI-compatible API)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.prompt = prompt
    
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
                            {"type": "text", "text": self.prompt},
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
        
    def classify_caption(self, caption_data: str) -> Tuple[Optional[str], str]:
        """
        Query OpenRouter API for image classification.
        
        Returns: (predicted_code, full_response)
        """
        try:
            # Make vision API call with text prompt and image
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt},
                            {
                                "type": "text",
                                "text": caption_data
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
