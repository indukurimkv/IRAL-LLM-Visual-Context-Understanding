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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_images_list(self) -> List[Dict]:
        """Fetch list of all images in the project using the search endpoint."""
        try:
            # Use the project-level search endpoint with API key in query parameter
            search_url = f"{self.base_url}/search?api_key={self.api_key}"
            all_images = []
            limit = 100  # Number of images per page
            offset = 0
            
            while True:
                # POST request to search endpoint with in_dataset filter
                # Body only contains in_dataset as specified
                payload = {
                    "in_dataset": True
                }
                # Always include limit parameter, add offset for subsequent pages
                paginated_url = f"{search_url}&limit={limit}"
                if offset > 0:
                    paginated_url += f"&offset={offset}"
                
                response = requests.post(
                    paginated_url, 
                    headers={"Content-Type": "application/json"}, 
                    json=payload,
                    timeout=30
                )
                # Raise exception if HTTP status indicates error
                response.raise_for_status()
                # Parse JSON response
                data = response.json()
                # Extract images from results field
                images = data.get("results", [])
                if not images:
                    # No more images to fetch
                    break
                all_images.extend(images)
                
                # Check pagination - get total
                total = data.get("total", 0)
                
                # Check if we've fetched all images
                if total > 0 and len(all_images) >= total:
                    # Fetched all images
                    break
                if len(images) < limit:
                    # Last page, no more images
                    break
                # Move to next page
                offset += limit
            
            return all_images
        except requests.exceptions.RequestException as e:
            print(f"Error fetching images list: {e}")
            raise
    
    def get_image_metadata(self, image_id: str) -> Dict:
        """Fetch full metadata for a specific image including annotations."""
        # Construct URL for specific image tags endpoint with API key in query parameter
        image_url = f"{self.base_url}/images/{image_id}?api_key={self.api_key}"
        try:
            # Get request to fetch detailed metadata including annotations
            response = requests.get(
                image_url, 
                headers={"Content-Type": "application/json"}, 
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            # Response is in format {"image": {...}}, extract the image object
            return data.get("image", data)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image metadata for {image_id}: {e}")
            raise
    
    def download_image(self, image_url: str) -> bytes:
        """Download image from URL."""
        try:
            # Download raw image bytes from URL
            response = requests.get(image_url, timeout=30)
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
        Extract prefix code from annotation.textAnnotations[0].prefix.
        
        Returns the 2-digit code (format: AH) or None if not found.
        """
        try:
            # Check if annotation exists
            if "annotation" not in annotation_data:
                return None
            
            # Navigate to annotation object
            annotation = annotation_data.get("annotation", {})
            if not annotation:
                return None
            
            # Extract textAnnotations array
            text_annotations = annotation.get("textAnnotations", [])
            if not text_annotations:
                return None
            
            # Get prefix code from first text annotation (format: AH where A=anomaly, H=hazard)
            prefix = text_annotations[0].get("prefix")
            return prefix if prefix else None
            
        except (KeyError, IndexError) as e:
            # Handle missing keys or empty lists gracefully
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
            
            # Download image - use original URL from urls.original, fallback to constructed download URL
            image_url = None
            if "urls" in image_metadata and "original" in image_metadata["urls"]:
                image_url = image_metadata["urls"]["original"]
            elif "url" in image_metadata:
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
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score using standard formulas.
        
        Uses macro-averaging: calculates precision/recall per class, then averages.
        Filters out results where model_prediction is None.
        
        Returns:
            Dictionary with 'precision', 'recall', and 'f1_score' keys
        """
        # Filter out results with null predictions
        valid_results = [r for r in self.results if r.get("model_prediction") is not None]
        
        if not valid_results:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
        
        # Get all unique classes
        all_classes = set()
        for r in valid_results:
            all_classes.add(r.get("ground_truth"))
            all_classes.add(r.get("model_prediction"))
        all_classes = sorted(list(all_classes))
        
        # Calculate precision and recall per class
        precisions = []
        recalls = []
        
        for cls in all_classes:
            # True Positives: predicted as cls and actually cls
            tp = sum(1 for r in valid_results 
                     if r.get("ground_truth") == cls and r.get("model_prediction") == cls)
            
            # False Positives: predicted as cls but actually not cls
            fp = sum(1 for r in valid_results 
                     if r.get("ground_truth") != cls and r.get("model_prediction") == cls)
            
            # False Negatives: actually cls but predicted as something else
            fn = sum(1 for r in valid_results 
                     if r.get("ground_truth") == cls and r.get("model_prediction") != cls)
            
            # Calculate precision for this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(precision)
            
            # Calculate recall for this class
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)
        
        # Macro-averaging: average across all classes
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        
        # Calculate F1 score from averaged precision and recall
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            "precision": round(avg_precision, 4),
            "recall": round(avg_recall, 4),
            "f1_score": round(f1_score, 4)
        }
    
    def generate_confusion_matrix(self, output_file: str) -> str:
        """
        Generate and save a confusion matrix visualization.
        
        Args:
            output_file: Path to save the confusion matrix image (PNG format)
        
        Returns:
            Path to the saved confusion matrix file
        """
        # Filter out results with null predictions
        valid_results = [r for r in self.results if r.get("model_prediction") is not None]
        
        if not valid_results:
            print("Warning: No valid predictions to generate confusion matrix")
            return ""
        
        # Extract ground truth and predictions
        ground_truth = [r.get("ground_truth") for r in valid_results]
        predictions = [r.get("model_prediction") for r in valid_results]
        
        # Define all possible classes and their labels
        class_codes = ["00", "01", "10", "11"]
        class_labels = {
            "00": "Neither",
            "01": "Hazard",
            "10": "Anomaly",
            "11": "Both"
        }
        
        # Create label lists for display
        display_labels = [class_labels[code] for code in class_codes]
        
        # Generate confusion matrix using sklearn
        cm = confusion_matrix(
            ground_truth,
            predictions,
            labels=class_codes
        )
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=display_labels,
               yticklabels=display_labels,
               title='Confusion Matrix',
               ylabel='Ground Truth',
               xlabel='Predicted')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations in each cell
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        # Save the figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {output_file}")
        return output_file
    
    def save_results(self, output_file: str, confusion_matrix_file: Optional[str] = None):
        """Save results to JSON file."""
        # Calculate statistics for summary
        total_images = len(self.results) + len(self.errors)
        successful = len(self.results)
        failed = len(self.errors)
        
        # Calculate accuracy: percentage of correct predictions
        matches = sum(1 for r in self.results if r.get("match", False))
        accuracy = matches / successful if successful > 0 else 0.0
        
        # Calculate precision, recall, and F1 score
        metrics = self.calculate_metrics()
        
        # Generate confusion matrix
        if confusion_matrix_file is None:
            # Default confusion matrix filename based on output file
            base_name = os.path.splitext(output_file)[0]
            confusion_matrix_file = f"{base_name}_confusion_matrix.png"
        
        confusion_matrix_path = self.generate_confusion_matrix(confusion_matrix_file)
        
        # Structure output data with results, errors, and summary statistics
        output_data = {
            "results": self.results,  # All successful classifications
            "errors": self.errors,  # All processing errors
            "summary": {
                "total_images": total_images,
                "successful": successful,
                "failed": failed,
                "accuracy": round(accuracy, 4),  # Rounded to 4 decimal places
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "confusion_matrix_file": confusion_matrix_path if confusion_matrix_path else None
            }
        }
        
        # Write results to JSON file with pretty formatting
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        print(f"Summary: {successful} successful, {failed} failed, accuracy: {accuracy:.2%}")
        print(f"Metrics: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")


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
    parser.add_argument(
        "--confusion-matrix-output",
        default=None,
        help="Output path for confusion matrix image (default: <output_file>_confusion_matrix.png)"
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
    processor.save_results(args.output, args.confusion_matrix_output)


if __name__ == "__main__":
    main()
