"""Mock client for simulating OpenRouter API responses without making actual API calls.

This module provides a MockOpenRouterClient class that mimics the behavior of
OpenRouterClient for testing purposes. It generates random classification codes
without incurring API costs.
"""

import random
import re
from typing import Optional, Tuple


class MockOpenRouterClient:
    """Mock client for simulating OpenRouter API responses.
    
    This class provides the same interface as OpenRouterClient but generates
    random classification codes instead of making actual API calls. Useful for
    testing the full workflow without spending API credits.
    
    Args:
        seed: Optional integer seed for random number generator. If provided,
              results will be deterministic and reproducible.
    """
    
    # Explanation mapping for each code
    CODE_EXPLANATIONS = {
        "00": "Safe: No anomalies or hazards detected",
        "01": "Dangerous: Hazard detected but no anomalies",
        "10": "Anomalous: Anomaly detected but no hazards",
        "11": "Anomalous Dangerous: Both anomaly and hazard detected"
    }
    
    # Valid classification codes
    VALID_CODES = ["00", "01", "10", "11"]
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize mock client with optional random seed."""
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.model = "mock-model"  # Placeholder model name
    
    def classify_image(self, image_data: bytes, image_format: str = "png") -> Tuple[Optional[str], str]:
        """
        Simulate image classification by generating a random code.
        
        This method mimics the behavior of OpenRouterClient.classify_image()
        but generates random responses instead of calling the API.
        
        Args:
            image_data: Image bytes (not used in mock mode, but kept for interface compatibility)
            image_format: Image format string (not used in mock mode)
        
        Returns:
            Tuple of (predicted_code, full_response) matching real API format
        """
        # Generate random classification code
        predicted_code = random.choice(self.VALID_CODES)
        
        # Format response string to match real API format: "<code>: <explanation>"
        explanation = self.CODE_EXPLANATIONS[predicted_code]
        full_response = f"{predicted_code}: {explanation}"
        
        return predicted_code, full_response
    
    @staticmethod
    def _extract_code(response: str) -> Optional[str]:
        """Extract 2-digit code from response text (same as real client)."""
        # Use regex to find first 2-digit number (word boundaries ensure exact match)
        match = re.search(r'\b\d{2}\b', response)
        if match:
            # Return the matched 2-digit code
            return match.group(0)
        return None
