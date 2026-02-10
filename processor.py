"""Main processor for classification workflow."""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

from annotation_parser import AnnotationParser
from metrics import calculate_metrics
from openrouter_client import OpenRouterClient
from roboflow_client import RoboflowAPIClient
from visualization import generate_confusion_matrix


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
    
    def process_image(self, image_metadata: Dict, image_data: bytes = None) -> Optional[Dict]:
        """Process a single image: extract annotation, download (if needed), classify.

        If `image_data` is provided, it will be used directly and no download
        from Roboflow will be attempted. This enables preloading images once
        and reusing them for multiple model queries.
        """
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
            
            # Download image with retry logic if not provided by caller
            if image_data is None:
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

    def process_preloaded_images(self, preloaded_images: List[Dict], progress_callback=None):
        """Process a list of preloaded images.

        Each item in `preloaded_images` should be a dict with keys:
            - "metadata": the image metadata dict (same format as get_image_metadata)
            - "image_data": raw bytes of the image

        This method avoids any Roboflow API calls and is intended to be used
        when images have been fetched once and should be classified by
        multiple different models.
        """
        total = len(preloaded_images)
        for idx, item in enumerate(preloaded_images, 1):
            metadata = item.get("metadata")
            image_data = item.get("image_data")
            if not metadata:
                print(f"Skipping preloaded item without metadata: {item}")
                continue

            print(f"Processing {idx}/{total}: {metadata.get('name', 'unknown')}")
            try:
                result = self.process_image(metadata, image_data=image_data)
                if result:
                    self.results.append(result)
            except Exception as e:
                error_msg = f"Error processing preloaded image {metadata.get('name', 'unknown')}: {e}"
                print(error_msg)
                self.errors.append({
                    "image_name": metadata.get('name', 'unknown'),
                    "error": str(e)
                })

            if progress_callback:
                progress_callback(idx, total)

            time.sleep(0.5)
    
    def save_results(self, output_file: str, confusion_matrix_file: Optional[str] = None, model_name: Optional[str] = None):
        """Save results to JSON file."""
        # Calculate statistics for summary
        total_images = len(self.results) + len(self.errors)
        successful = len(self.results)
        failed = len(self.errors)
        
        # Calculate accuracy: percentage of correct predictions
        matches = sum(1 for r in self.results if r.get("match", False))
        accuracy = matches / successful if successful > 0 else 0.0
        
        # Calculate precision, recall, and F1 score using standalone function
        metrics = calculate_metrics(self.results)
        
        # Generate confusion matrix using standalone function
        if confusion_matrix_file is None:
            # Default confusion matrix filename based on output file
            base_name = os.path.splitext(output_file)[0]
            confusion_matrix_file = f"{base_name}_confusion_matrix.png"
        
        confusion_matrix_path = generate_confusion_matrix(self.results, confusion_matrix_file, model_name=model_name)
        
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
