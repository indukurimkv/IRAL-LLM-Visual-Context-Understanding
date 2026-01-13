#!/usr/bin/env python3
"""
Roboflow to OpenRouter Classification Script

This script fetches images and annotations from Roboflow via REST API,
queries OpenRouter's Claude 3.5 Sonnet Vision model for AH classification,
and stores results with ground truth annotations.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from annotation_parser import AnnotationParser
from mock_openrouter_client import MockOpenRouterClient
from openrouter_client import OpenRouterClient
from processor import ClassificationProcessor
from roboflow_client import RoboflowAPIClient

# Load environment variables from .env file
load_dotenv()


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
        help="OpenRouter model(s) to use. Comma-separated list supported (default: anthropic/claude-3.5-sonnet)"
    )
    parser.add_argument(
        "--confusion-matrix-output",
        default=None,
        help="Output path for confusion matrix image (default: <output_file>_confusion_matrix.png)"
    )
    parser.add_argument(
        "--class-max",
        type=int,
        default=None,
        help="Maximum images to download per class (00, 01, 10, 11). If omitted, no per-class limit"
    )
    parser.add_argument(
        "--mock-mode",
        action="store_true",
        help="Enable mock mode: simulate OpenRouter API responses without making actual API calls (saves credits)"
    )
    parser.add_argument(
        "--mock-seed",
        type=int,
        default=None,
        help="Random seed for mock mode (for reproducible results). Only used when --mock-mode is enabled"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Validate that API keys are provided (either via CLI, .env, or env vars)
    if not args.roboflow_api_key:
        print("Error: Roboflow API key is required. Set ROBOFLOW_API_KEY in .env file or use --roboflow-api-key")
        sys.exit(1)
    
    # OpenRouter API key is only required when not in mock mode
    if not args.mock_mode and not args.openrouter_api_key:
        print("Error: OpenRouter API key is required when not in mock mode. Set OPENROUTER_API_KEY in .env file or use --openrouter-api-key")
        sys.exit(1)
    
    # Validate that workspace and project IDs are provided
    if not args.workspace_id:
        print("Error: Roboflow workspace ID is required. Set ROBOFLOW_WORKSPACE_ID in .env file or use --workspace-id")
        sys.exit(1)
    
    if not args.project_id:
        print("Error: Roboflow project ID is required. Set ROBOFLOW_PROJECT_ID in .env file or use --project-id")
        sys.exit(1)

    if args.class_max is not None and args.class_max <= 0:
        print("Error: --class-max must be a positive integer when provided")
        sys.exit(1)
    
    # Initialize API clients with credentials
    print("Initializing clients...")
    roboflow_client = RoboflowAPIClient(
        args.roboflow_api_key,
        args.workspace_id,
        args.project_id
    )
    # Parse models list (support comma-separated for backward compatibility)
    models = [m.strip() for m in args.model.split(',') if m.strip()]

    # Initialize processors for all models
    processors = []
    print(f"Initializing processors for models: {models}")
    
    for model in models:
        model_safe = model.replace('/', '_')
        if args.mock_mode:
            openrouter_client = MockOpenRouterClient(seed=args.mock_seed)
        else:
            openrouter_client = OpenRouterClient(args.openrouter_api_key, model)
            
        processor = ClassificationProcessor(roboflow_client, openrouter_client)
        processors.append({
            "model": model,
            "model_safe": model_safe,
            "processor": processor,
            "openrouter_client": openrouter_client
        })
        
    if args.mock_mode:
        print("MOCK MODE ENABLED: Simulating OpenRouter API responses")
        if args.mock_seed is not None:
            print(f"Using random seed: {args.mock_seed}")

    print("Starting processing. Fetching images one by one...")
    
    class_max = args.class_max
    class_codes = ["00", "01", "10", "11"]
    class_counts = {code: 0 for code in class_codes}
    annotation_parser = AnnotationParser()

    def limits_reached() -> bool:
        return class_max is not None and all(count >= class_max for count in class_counts.values())

    stop_due_to_limits = False

    try:
        # Iterate through images lazily
        for idx, img_summary in enumerate(roboflow_client.yield_images(), 1):
            if limits_reached():
                print(f"All classes have reached the limit of {class_max}. Stopping early and saving results...")
                break

            image_id = img_summary.get("id")
            if not image_id:
                print(f"Skipping image without ID: {img_summary}")
                continue

            print(f"\nProcessing image {idx}: {img_summary.get('name', image_id)}")
            
            try:
                # 1. Fetch metadata
                metadata = roboflow_client.get_image_metadata(image_id)
                ground_truth = annotation_parser.extract_prefix_code(metadata)

                if class_max is not None:
                    if ground_truth is None:
                        print(f"Skipping {img_summary.get('name', image_id)}: No annotation found for class limit check")
                        continue

                    # Include any unexpected class in tracking so limits still apply uniformly
                    if ground_truth not in class_counts:
                        class_counts[ground_truth] = 0

                    if class_counts[ground_truth] >= class_max:
                        print(f"Skipping {img_summary.get('name', image_id)}: Class {ground_truth} reached limit {class_max}")
                        continue
                
                # 2. Determine URL and download image ONCE
                image_url = None
                if "urls" in metadata and "original" in metadata["urls"]:
                    image_url = metadata["urls"]["original"]
                elif "url" in metadata:
                    image_url = metadata["url"]
                else:
                    image_url = f"{roboflow_client.base_url}/images/{image_id}/download"

                image_data = roboflow_client.download_image(image_url)

                if class_max is not None and ground_truth is not None:
                    class_counts[ground_truth] += 1
                
                # 3. Process with ALL models
                for p_data in processors:
                    model = p_data["model"]
                    processor = p_data["processor"]
                    # print(f"  Querying {model}...")
                    try:
                        result = processor.process_image(metadata, image_data=image_data)
                        if result:
                            processor.results.append(result)
                    except Exception as e:
                        print(f"  Error querying {model} for {image_id}: {e}")

                if limits_reached():
                    stop_due_to_limits = True
                    print(f"Reached class limit {class_max} for all classes after {img_summary.get('name', image_id)}. Stopping...\n")
                    break
                        
            except Exception as e:
                print(f"Error processing image {image_id}: {e}")
                continue

        if stop_due_to_limits:
            print("Class limits satisfied; proceeding to save results.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")
    except Exception as e:
        print(f"Fatal error during processing: {e}")

    # Save results for all processors
    print("\nSaving results...")
    for p_data in processors:
        model_safe = p_data["model_safe"]
        processor = p_data["processor"]
        
        # Define results directory
        results_dir = "results"
        model_dir = os.path.join(results_dir, model_safe)
        os.makedirs(model_dir, exist_ok=True)
        
        # Construct output paths inside the model directory
        output_path = os.path.join(model_dir, args.output)
        
        confusion_path = None
        if args.confusion_matrix_output:
            confusion_path = os.path.join(model_dir, args.confusion_matrix_output)
        
        print(f"Saving results for {p_data['model']} to {model_dir}...")
        processor.save_results(output_path, confusion_path)


if __name__ == "__main__":
    main()
