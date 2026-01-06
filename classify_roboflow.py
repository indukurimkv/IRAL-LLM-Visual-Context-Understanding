#!/usr/bin/env python3
"""
Roboflow to OpenRouter Classification Script

This script fetches images and annotations from Roboflow via REST API,
queries OpenRouter's Claude 3.5 Sonnet Vision model for AH classification,
and stores results with ground truth annotations.
"""

import argparse
import json
import os
import sys
import time
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from io import BytesIO

import requests
from openai import OpenAI
from PIL import Image


class RoboflowAPIClient:
    """Client for interacting with Roboflow REST API."""
    
    def __init__(self, api_key: str, workspace_id: str, project_id: str):
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.project_id = project_id
        self.base_url = f"https://api.roboflow.com/{workspace_id}/{project_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def get_images_list(self) -> List[Dict]:
        """Fetch list of all images in the project."""
        try:
            response = requests.get(self.base_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching images list: {e}")
            raise
    
    def get_image_metadata(self, image_id: str) -> Dict:
        """Fetch full metadata for a specific image including annotations."""
        image_url = f"{self.base_url}/images/{image_id}"
        try:
            response = requests.get(image_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image metadata for {image_id}: {e}")
            raise
    
    def download_image(self, image_url: str) -> bytes:
        """Download image from URL."""
        try:
            response = requests.get(image_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {image_url}: {e}")
            raise


class AnnotationParser:
    """Parser for extracting prefix codes from Roboflow annotations."""
    
    @staticmethod
    def extract_prefix_code(annotation_data: Dict) -> Optional[str]:
        """
        Extract prefix code from annotations["hazards-and-anomalies"]["converted"].
        
        Returns the 2-digit code (format: AH) or None if not found.
        """
        try:
            # Check if hazards-and-anomalies annotation exists
            if "annotations" not in annotation_data:
                return None
            
            annotations = annotation_data.get("annotations", {})
            hazards_annotation = annotations.get("hazards-and-anomalies", {})
            
            if not hazards_annotation:
                return None
            
            # The "converted" field contains a JSON string
            converted_str = hazards_annotation.get("converted")
            if not converted_str:
                return None
            
            # Parse the JSON string
            converted_data = json.loads(converted_str)
            
            # Extract prefix from textAnnotations
            text_annotations = converted_data.get("textAnnotations", [])
            if not text_annotations:
                return None
            
            prefix = text_annotations[0].get("prefix")
            return prefix if prefix else None
            
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            print(f"Error parsing annotation: {e}")
            return None


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        self.api_key = api_key
        self.model = model
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
            # Encode image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            image_data_url = f"data:image/{image_format};base64,{image_base64}"
            
            # Create the prompt
            prompt = (
                "Analyze this image and provide a 2-digit numeric code in format AH "
                "where A=anomaly (0 or 1) and H=hazard (0 or 1). "
                "Only respond with the 2-digit code."
            )
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url}
                            }
                        ]
                    }
                ],
                max_tokens=10
            )
            
            full_response = response.choices[0].message.content.strip()
            
            # Extract 2-digit code from response
            predicted_code = self._extract_code(full_response)
            
            return predicted_code, full_response
            
        except Exception as e:
            print(f"Error querying OpenRouter API: {e}")
            return None, str(e)
    
    @staticmethod
    def _extract_code(response: str) -> Optional[str]:
        """Extract 2-digit code from model response."""
        # Look for 2-digit number in the response
        import re
        match = re.search(r'\b\d{2}\b', response)
        if match:
            return match.group(0)
        return None


class ClassificationProcessor:
    """Main processor for classification workflow."""
    
    def __init__(self, roboflow_client: RoboflowAPIClient, 
                 openrouter_client: OpenRouterClient,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        self.roboflow_client = roboflow_client
        self.openrouter_client = openrouter_client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.results = []
        self.errors = []
    
    def process_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Retry attempt {attempt + 1}/{self.max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
    
    def process_image(self, image_metadata: Dict) -> Optional[Dict]:
        """Process a single image: extract annotation, download, classify."""
        try:
            image_name = image_metadata.get("name", "unknown")
            image_id = image_metadata.get("id")
            
            # Extract ground truth prefix code
            parser = AnnotationParser()
            ground_truth = parser.extract_prefix_code(image_metadata)
            
            if ground_truth is None:
                print(f"Skipping {image_name}: No annotation found")
                return None
            
            # Download image
            # Try to get image URL from metadata or construct it
            image_url = None
            if "url" in image_metadata:
                image_url = image_metadata["url"]
            else:
                # Construct URL if not provided
                image_url = f"{self.roboflow_client.base_url}/images/{image_id}/download"
            
            image_data = self.process_with_retry(
                self.roboflow_client.download_image, image_url
            )
            
            # Determine image format from extension
            image_format = image_name.split('.')[-1].lower() if '.' in image_name else "png"
            
            # Query OpenRouter
            predicted_code, model_response = self.process_with_retry(
                self.openrouter_client.classify_image, image_data, image_format
            )
            
            if predicted_code is None:
                print(f"Warning: Could not extract code from model response for {image_name}")
            
            result = {
                "image_name": image_name,
                "image_id": image_id,
                "ground_truth": ground_truth,
                "model_prediction": predicted_code,
                "model_response": model_response,
                "timestamp": datetime.now().isoformat(),
                "match": ground_truth == predicted_code if predicted_code else False
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing image {image_metadata.get('name', 'unknown')}: {e}"
            print(error_msg)
            self.errors.append({
                "image_name": image_metadata.get("name", "unknown"),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return None
    
    def process_all_images(self, progress_callback=None):
        """Process all images in the project."""
        print("Fetching images list from Roboflow...")
        images_list = self.process_with_retry(self.roboflow_client.get_images_list)
        
        print(f"Found {len(images_list)} images. Processing...")
        
        for idx, image_summary in enumerate(images_list, 1):
            print(f"Processing {idx}/{len(images_list)}: {image_summary.get('name', 'unknown')}")
            
            # Fetch full metadata
            image_id = image_summary.get("id")
            if not image_id:
                print(f"Skipping image without ID: {image_summary}")
                continue
            
            image_metadata = self.process_with_retry(
                self.roboflow_client.get_image_metadata, image_id
            )
            
            # Process image
            result = self.process_image(image_metadata)
            if result:
                self.results.append(result)
            
            # Progress callback
            if progress_callback:
                progress_callback(idx, len(images_list))
            
            # Rate limiting - small delay between requests
            time.sleep(0.5)
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        total_images = len(self.results) + len(self.errors)
        successful = len(self.results)
        failed = len(self.errors)
        
        # Calculate accuracy
        matches = sum(1 for r in self.results if r.get("match", False))
        accuracy = matches / successful if successful > 0 else 0.0
        
        output_data = {
            "results": self.results,
            "errors": self.errors,
            "summary": {
                "total_images": total_images,
                "successful": successful,
                "failed": failed,
                "accuracy": round(accuracy, 4)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        print(f"Summary: {successful} successful, {failed} failed, accuracy: {accuracy:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify Roboflow images using OpenRouter API"
    )
    parser.add_argument(
        "--workspace-id",
        required=True,
        help="Roboflow workspace ID"
    )
    parser.add_argument(
        "--project-id",
        required=True,
        help="Roboflow project ID"
    )
    parser.add_argument(
        "--roboflow-api-key",
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)"
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--output",
        default="classification_results.json",
        help="Output JSON file path (default: classification_results.json)"
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="OpenRouter model to use (default: anthropic/claude-3.5-sonnet)"
    )
    
    args = parser.parse_args()
    
    # Validate API keys
    if not args.roboflow_api_key:
        print("Error: Roboflow API key is required. Set ROBOFLOW_API_KEY env var or use --roboflow-api-key")
        sys.exit(1)
    
    if not args.openrouter_api_key:
        print("Error: OpenRouter API key is required. Set OPENROUTER_API_KEY env var or use --openrouter-api-key")
        sys.exit(1)
    
    # Initialize clients
    print("Initializing clients...")
    roboflow_client = RoboflowAPIClient(
        args.roboflow_api_key,
        args.workspace_id,
        args.project_id
    )
    
    openrouter_client = OpenRouterClient(args.openrouter_api_key, args.model)
    
    # Create processor
    processor = ClassificationProcessor(roboflow_client, openrouter_client)
    
    # Process all images
    try:
        processor.process_all_images()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    
    # Save results
    processor.save_results(args.output)


if __name__ == "__main__":
    main()
