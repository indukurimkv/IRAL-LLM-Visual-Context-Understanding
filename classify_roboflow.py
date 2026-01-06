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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class RoboflowAPIClient:
    """Client for interacting with Roboflow REST API."""
    
    def __init__(self, api_key: str, workspace_id: str, project_id: str):
        # Store API credentials and identifiers
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.project_id = project_id
        # Construct base URL for API requests
        self.base_url = f"https://api.roboflow.com/{workspace_id}/{project_id}"
        # Set authorization header for authenticated requests
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def get_images_list(self) -> List[Dict]:
        """Fetch list of all images in the project."""
        try:
            # GET request to base URL returns list of all images
            response = requests.get(self.base_url, headers=self.headers, timeout=30)
            # Raise exception if HTTP status indicates error
            response.raise_for_status()
            # Parse and return JSON response
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching images list: {e}")
            raise
    
    def get_image_metadata(self, image_id: str) -> Dict:
        """Fetch full metadata for a specific image including annotations."""
        # Construct URL for specific image endpoint
        image_url = f"{self.base_url}/images/{image_id}"
        try:
            # Fetch detailed metadata including annotations
            response = requests.get(image_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image metadata for {image_id}: {e}")
            raise
    
    def download_image(self, image_url: str) -> bytes:
        """Download image from URL."""
        try:
            # Download raw image bytes from URL
            response = requests.get(image_url, headers=self.headers, timeout=30)
            response.raise_for_status()
            # Return binary content (image data)
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
            
            # Navigate to hazards-and-anomalies annotation
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
            
            # Extract prefix from first text annotation (ground truth code)
            text_annotations = converted_data.get("textAnnotations", [])
            if not text_annotations:
                return None
            
            # Get prefix code (format: AH where A=anomaly, H=hazard)
            prefix = text_annotations[0].get("prefix")
            return prefix if prefix else None
            
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            # Handle missing keys, invalid JSON, or empty lists gracefully
            print(f"Error parsing annotation: {e}")
            return None


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
            
            # Create classification prompt specifying expected output format
            prompt = (
                "Analyze this image and provide a 2-digit numeric code in format AH "
                "where A=anomaly (0 or 1) and H=hazard (0 or 1). "
                "Only respond with the 2-digit code."
            )
            
            # Make vision API call with text prompt and image
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
                max_tokens=10  # Limit response length since we only need 2 digits
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
        import re
        match = re.search(r'\b\d{2}\b', response)
        if match:
            # Return the matched 2-digit code
            return match.group(0)
        return None


class ClassificationProcessor:
    """Main processor for classification workflow."""
    
    def __init__(self, roboflow_client: RoboflowAPIClient, 
                 openrouter_client: OpenRouterClient,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        # Store API clients for Roboflow and OpenRouter
        self.roboflow_client = roboflow_client
        self.openrouter_client = openrouter_client
        # Configure retry behavior for failed requests
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        # Store successful classification results
        self.results = []
        # Store errors encountered during processing
        self.errors = []
    
    def process_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                # Try executing the function
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                # If not last attempt, wait and retry with exponential backoff
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, etc.
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Retry attempt {attempt + 1}/{self.max_retries} after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Last attempt failed, re-raise exception
                    raise
    
    def process_image(self, image_metadata: Dict) -> Optional[Dict]:
        """Process a single image: extract annotation, download, classify."""
        try:
            # Extract image identifier and name from metadata
            image_name = image_metadata.get("name", "unknown")
            image_id = image_metadata.get("id")
            
            # Extract ground truth prefix code from Roboflow annotations
            parser = AnnotationParser()
            ground_truth = parser.extract_prefix_code(image_metadata)
            
            # Skip images without annotations (no ground truth to compare)
            if ground_truth is None:
                print(f"Skipping {image_name}: No annotation found")
                return None
            
            # Download image - try direct URL first, fallback to constructed download URL
            image_url = None
            if "url" in image_metadata:
                image_url = image_metadata["url"]
            else:
                # Construct download URL using image ID
                image_url = f"{self.roboflow_client.base_url}/images/{image_id}/download"
            
            # Download image with retry logic
            image_data = self.process_with_retry(
                self.roboflow_client.download_image, image_url
            )
            
            # Determine image format from file extension (for base64 encoding)
            image_format = image_name.split('.')[-1].lower() if '.' in image_name else "png"
            
            # Query OpenRouter API for classification with retry logic
            predicted_code, model_response = self.process_with_retry(
                self.openrouter_client.classify_image, image_data, image_format
            )
            
            # Warn if code extraction failed
            if predicted_code is None:
                print(f"Warning: Could not extract code from model response for {image_name}")
            
            # Build result dictionary with all relevant information
            result = {
                "image_name": image_name,
                "image_id": image_id,
                "ground_truth": ground_truth,  # Expected code from annotations
                "model_prediction": predicted_code,  # Model's predicted code
                "model_response": model_response,  # Full model response text
                "timestamp": datetime.now().isoformat(),  # Processing timestamp
                "match": ground_truth == predicted_code if predicted_code else False  # Accuracy check
            }
            
            return result
            
        except Exception as e:
            # Log error and store in errors list for reporting
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
        # Get list of all images in the project (with retry on failure)
        images_list = self.process_with_retry(self.roboflow_client.get_images_list)
        
        print(f"Found {len(images_list)} images. Processing...")
        
        # Process each image sequentially
        for idx, image_summary in enumerate(images_list, 1):
            print(f"Processing {idx}/{len(images_list)}: {image_summary.get('name', 'unknown')}")
            
            # Fetch full metadata including annotations (needed for ground truth)
            image_id = image_summary.get("id")
            if not image_id:
                print(f"Skipping image without ID: {image_summary}")
                continue
            
            # Get detailed metadata with retry logic
            image_metadata = self.process_with_retry(
                self.roboflow_client.get_image_metadata, image_id
            )
            
            # Process image: download, classify, and compare
            result = self.process_image(image_metadata)
            if result:
                # Store successful result
                self.results.append(result)
            
            # Optional progress callback for external monitoring
            if progress_callback:
                progress_callback(idx, len(images_list))
            
            # Rate limiting - small delay between requests to avoid API throttling
            time.sleep(0.5)
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        # Calculate statistics for summary
        total_images = len(self.results) + len(self.errors)
        successful = len(self.results)
        failed = len(self.errors)
        
        # Calculate accuracy: percentage of correct predictions
        matches = sum(1 for r in self.results if r.get("match", False))
        accuracy = matches / successful if successful > 0 else 0.0
        
        # Structure output data with results, errors, and summary statistics
        output_data = {
            "results": self.results,  # All successful classifications
            "errors": self.errors,  # All processing errors
            "summary": {
                "total_images": total_images,
                "successful": successful,
                "failed": failed,
                "accuracy": round(accuracy, 4)  # Rounded to 4 decimal places
            }
        }
        
        # Write results to JSON file with pretty formatting
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        print(f"Summary: {successful} successful, {failed} failed, accuracy: {accuracy:.2%}")


def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Classify Roboflow images using OpenRouter API"
    )
    # Arguments for Roboflow project identification (can be from .env or CLI)
    parser.add_argument(
        "--workspace-id",
        default=os.getenv("ROBOFLOW_WORKSPACE_ID"),
        help="Roboflow workspace ID (or set ROBOFLOW_WORKSPACE_ID in .env file)"
    )
    parser.add_argument(
        "--project-id",
        default=os.getenv("ROBOFLOW_PROJECT_ID"),
        help="Roboflow project ID (or set ROBOFLOW_PROJECT_ID in .env file)"
    )
    # API keys can be provided via CLI, .env file, or environment variables
    parser.add_argument(
        "--roboflow-api-key",
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY in .env file)"
    )
    parser.add_argument(
        "--openrouter-api-key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (or set OPENROUTER_API_KEY in .env file)"
    )
    # Optional arguments with defaults
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
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Validate that API keys are provided (either via CLI, .env, or env vars)
    if not args.roboflow_api_key:
        print("Error: Roboflow API key is required. Set ROBOFLOW_API_KEY in .env file or use --roboflow-api-key")
        sys.exit(1)
    
    if not args.openrouter_api_key:
        print("Error: OpenRouter API key is required. Set OPENROUTER_API_KEY in .env file or use --openrouter-api-key")
        sys.exit(1)
    
    # Validate that workspace and project IDs are provided
    if not args.workspace_id:
        print("Error: Roboflow workspace ID is required. Set ROBOFLOW_WORKSPACE_ID in .env file or use --workspace-id")
        sys.exit(1)
    
    if not args.project_id:
        print("Error: Roboflow project ID is required. Set ROBOFLOW_PROJECT_ID in .env file or use --project-id")
        sys.exit(1)
    
    # Initialize API clients with credentials
    print("Initializing clients...")
    roboflow_client = RoboflowAPIClient(
        args.roboflow_api_key,
        args.workspace_id,
        args.project_id
    )
    
    openrouter_client = OpenRouterClient(args.openrouter_api_key, args.model)
    
    # Create main processor that orchestrates the workflow
    processor = ClassificationProcessor(roboflow_client, openrouter_client)
    
    # Process all images in the project
    try:
        processor.process_all_images()
    except KeyboardInterrupt:
        # Handle user interruption gracefully - save partial results
        print("\nInterrupted by user. Saving partial results...")
    except Exception as e:
        # Handle fatal errors
        print(f"Fatal error: {e}")
        sys.exit(1)
    
    # Save all results and statistics to JSON file
    processor.save_results(args.output)


if __name__ == "__main__":
    main()
